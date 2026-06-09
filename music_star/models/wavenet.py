"""Conditional WaveNet decoder used for Music-STAR generation and training."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Conv1d):
    """One-dimensional convolution cropped to preserve causality.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int, optional
        Temporal kernel size.
    dilation : int, optional
        Dilation factor.
    **kwargs
        Additional arguments passed to :class:`torch.nn.Conv1d`.
    """

    def __init__(self, in_channels, out_channels, kernel_size=2, dilation=1, **kwargs):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            padding=dilation * (kernel_size - 1),
            dilation=dilation,
            **kwargs,
        )

    def forward(self, input):
        """Apply causal convolution.

        Parameters
        ----------
        input : torch.Tensor
            Input tensor with shape ``(batch, channels, time)``.

        Returns
        -------
        torch.Tensor
            Cropped convolution output.
        """

        out = super().forward(input)
        return out[:, :, : -self.padding[0]]


class WavenetLayer(nn.Module):
    """Single gated residual WaveNet layer.

    Parameters
    ----------
    residual_channels : int
        Number of residual channels.
    skip_channels : int
        Number of skip channels.
    cond_channels : int
        Number of conditioning channels.
    kernel_size : int, optional
        Causal convolution kernel size.
    dilation : int, optional
        Causal convolution dilation.
    """

    def __init__(self, residual_channels, skip_channels, cond_channels, kernel_size=2, dilation=1):
        super().__init__()

        self.causal = CausalConv1d(
            residual_channels,
            2 * residual_channels,
            kernel_size,
            dilation=dilation,
            bias=True,
        )
        self.condition = nn.Conv1d(cond_channels, 2 * residual_channels, kernel_size=1, bias=True)
        self.residual = nn.Conv1d(residual_channels, residual_channels, kernel_size=1, bias=True)
        self.skip = nn.Conv1d(residual_channels, skip_channels, kernel_size=1, bias=True)

    def _condition(self, x, c, f):
        c = f(c)
        x = x + c
        return x

    def forward(self, x, c=None):
        """Apply one WaveNet layer.

        Parameters
        ----------
        x : torch.Tensor
            Residual input with shape ``(batch, residual_channels, time)``.
        c : torch.Tensor | None, optional
            Conditioning tensor broadcastable to the residual time axis.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Residual and skip tensors.
        """

        x = self.causal(x)
        # biasing the layer data (x) by the projection (f) of the condition (c)
        if c is not None:
            x = self._condition(x, c, self.condition)

        assert x.size(1) % 2 == 0
        gate, output = x.chunk(2, 1)
        gate = torch.sigmoid(gate)
        output = torch.tanh(output)
        x = gate * output

        residual = self.residual(x)
        skip = self.skip(x)

        return residual, skip


class WaveNet(nn.Module):
    """Conditional WaveNet decoder for 8-bit mu-law waveform modeling.

    Parameters
    ----------
    args
        Namespace with ``blocks``, ``layers``, ``kernel_size``,
        ``skip_channels``, ``residual_channels``, and ``latent_d``.
    create_layers : bool, optional
        Whether to instantiate the residual WaveNet stack.
    shift_input : bool, optional
        Whether to shift teacher-forced inputs right during training.
    """

    def __init__(self, args, create_layers=True, shift_input=True):
        super().__init__()

        self.blocks = args.blocks
        self.layer_num = args.layers
        self.kernel_size = args.kernel_size
        self.skip_channels = args.skip_channels
        self.residual_channels = args.residual_channels
        self.cond_channels = args.latent_d
        self.classes = 256
        self.shift_input = shift_input

        if create_layers:
            layers = []
            for _ in range(self.blocks):
                for i in range(self.layer_num):
                    dilation = 2**i
                    layers.append(
                        WavenetLayer(
                            self.residual_channels,
                            self.skip_channels,
                            self.cond_channels,
                            self.kernel_size,
                            dilation,
                        )
                    )
            self.layers = nn.ModuleList(layers)

        self.first_conv = CausalConv1d(1, self.residual_channels, kernel_size=self.kernel_size)
        self.skip_conv = nn.Conv1d(self.residual_channels, self.skip_channels, kernel_size=1)
        self.condition = nn.Conv1d(self.cond_channels, self.skip_channels, kernel_size=1)
        self.fc = nn.Conv1d(self.skip_channels, self.skip_channels, kernel_size=1)
        self.logits = nn.Conv1d(self.skip_channels, self.classes, kernel_size=1)

    def _condition(self, x, c, f):
        c = f(c)
        x = x + c
        return x

    @staticmethod
    def _upsample_cond(x, c):
        bsz, channels, length = x.size()
        cond_bsz, cond_channels, cond_length = c.size()
        assert bsz == cond_bsz

        if c.size(2) != 1:
            # unsqueeze adds another dimension, repeat: repeat dim 0,1,2 for 1 time and repeat the last dim
            # values for length/cond_l times and the collapse the repeats into one dim as in the view
            c = c.unsqueeze(3).repeat(1, 1, 1, length // cond_length)
            c = c.view(bsz, cond_channels, length)

        return c

    @staticmethod
    def shift_right(x):
        """Shift an audio sequence by one sample for teacher forcing.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape ``(batch, channels, time)``.

        Returns
        -------
        torch.Tensor
            Shifted tensor with the same shape as ``x``.
        """

        x = F.pad(x, (1, 0))
        return x[:, :, :-1]

    def forward(self, x, c=None):
        """Predict mu-law logits from audio and optional latent conditioning.

        Parameters
        ----------
        x : torch.Tensor
            Input audio tensor with shape ``(batch, time)`` or
            ``(batch, 1, time)``.
        c : torch.Tensor | None, optional
            Latent conditioning with shape ``(batch, latent_d, cond_time)``.

        Returns
        -------
        torch.Tensor
            Logits with shape ``(batch, 256, time)``.
        """

        if x.dim() < 3:
            x = x.unsqueeze(1)
        if ("Half" not in x.type()) and ("Float" not in x.type()):
            x = x.float()

        # normalize the input in the [-0.5, 0.5]
        x = x / 255 - 0.5

        if self.shift_input:
            x = self.shift_right(x)

        # make c to match the dimensions of the input
        if c is not None:
            c = self._upsample_cond(x, c)

        residual = self.first_conv(x)
        skip = self.skip_conv(residual)

        for layer in self.layers:
            r, s = layer(residual, c)
            residual = residual + r
            skip = skip + s

        skip = F.relu(skip)
        skip = self.fc(skip)
        if c is not None:
            skip = self._condition(skip, c, self.condition)
        skip = F.relu(skip)
        skip = self.logits(skip)

        return skip
