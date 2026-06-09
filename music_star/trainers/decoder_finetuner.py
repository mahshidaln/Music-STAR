"""Decoder finetuning trainer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from music_star.checkpoints import load_checkpoint_state, load_saved_args
from music_star.data import Dataset
from music_star.models import WaveNet
from music_star.models.universal import cross_entropy_loss
from music_star.trainers.base import (
    BaseTrainer,
    data_paths,
    epoch_length,
    epochs,
    legacy_or_standard_encoder,
    to_device,
)
from music_star.utils import LossManager


class DecoderFinetuner(BaseTrainer):
    """Finetune a target WaveNet decoder with a frozen encoder.

    Parameters
    ----------
    args
        Namespace with data paths, checkpoint path, and optimizer settings.
    """

    checkpoint_keys = ("encoder_state", "decoder_state")

    def __init__(self, args: Any):
        super().__init__(args)
        self.args.n_datasets = len(data_paths(args))
        self.data = [Dataset(args, path) for path in data_paths(args)]
        self.recon_losses = [LossManager(f"recon {i}") for i in range(self.args.n_datasets)]
        self.eval_total = LossManager("eval total")

        pretrained_checkpoint = Path(
            getattr(
                args,
                "pretrained_checkpoint",
                Path(args.checkpoint_dir) / getattr(args, "model_file", "bestmodel_0.pth"),
            )
        )
        pretrained_dir = pretrained_checkpoint.parent
        model_args = load_saved_args(pretrained_dir)
        checkpoint = load_checkpoint_state(pretrained_dir, pretrained_checkpoint.name)

        self.encoder = legacy_or_standard_encoder(model_args).to(self.device)
        self.encoder.load_state_dict(checkpoint["encoder_state"], strict=True)
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

        self.decoder = WaveNet(model_args).to(self.device)
        self.decoder.load_state_dict(checkpoint["decoder_state"], strict=True)
        self.model_optimizer = optim.Adam(self.decoder.parameters(), lr=float(args.lr))
        self.lr_manager = optim.lr_scheduler.ExponentialLR(
            self.model_optimizer, float(getattr(args, "lr_decay", 0.98))
        )

    def train_batch(self, x: torch.Tensor, dataset_index: int) -> float:
        """Train one decoder-finetuning batch.

        Parameters
        ----------
        x : torch.Tensor
            Mu-law audio batch.
        dataset_index : int
            Domain index used for metric bookkeeping.

        Returns
        -------
        float
            Reconstruction loss.
        """

        x = to_device(x.float(), self.device)
        with torch.no_grad():
            z = self.encoder(x)
        y = self.decoder(x, z)
        recon_loss = cross_entropy_loss(y, x)
        loss = recon_loss.mean()

        self.model_optimizer.zero_grad()
        loss.backward()
        self._clip(self.decoder.parameters())
        self.model_optimizer.step()

        self.recon_losses[dataset_index].add(float(recon_loss.detach().mean().cpu()))
        return float(loss.detach().cpu())

    def train(self) -> None:
        """Run decoder finetuning and checkpointing."""

        best_eval = float("inf")
        for epoch in range(self.start_epoch, self.start_epoch + epochs(self.args)):
            self.decoder.train()
            self.encoder.eval()
            for meter in self.recon_losses:
                meter.reset()

            with tqdm(total=epoch_length(self.args), desc=f"Train epoch {epoch}") as train_enum:
                for batch_num in range(epoch_length(self.args)):
                    dataset_index = batch_num % self.args.n_datasets
                    x, _ = next(self.data[dataset_index].train_iter)
                    batch_loss = self.train_batch(x, dataset_index)
                    train_enum.set_description(f"Train (loss: {batch_loss:.2f}) epoch {epoch}")
                    train_enum.update()

            val_loss = self.evaluate(epoch)
            self.lr_manager.step()
            if val_loss < best_eval:
                self.save_model("bestmodel_0.pth")
                best_eval = val_loss
            self.save_model("lastmodel_0.pth")
            self._save_args(epoch)

    def evaluate(self, epoch: int) -> float:
        """Evaluate the finetuned decoder.

        Parameters
        ----------
        epoch : int
            Epoch number used for progress display.

        Returns
        -------
        float
            Mean validation reconstruction loss.
        """

        self.encoder.eval()
        self.decoder.eval()
        self.eval_total.reset()
        total_batches = max(int(np.ceil(epoch_length(self.args) / 10)), 1)
        with tqdm(total=total_batches, desc=f"Eval epoch {epoch}") as valid_enum, torch.no_grad():
            for batch_num in range(total_batches):
                dataset_index = batch_num % self.args.n_datasets
                x, _ = next(self.data[dataset_index].valid_iter)
                x = to_device(x.float(), self.device)
                z = self.encoder(x)
                y = self.decoder(x, z)
                loss = cross_entropy_loss(y, x).mean()
                self.eval_total.add(float(loss.detach().cpu()))
                valid_enum.update()
        return self.eval_total.epoch_mean()

    def save_model(self, filename: str) -> None:
        """Save frozen encoder state, finetuned decoder state, and optimizer.

        Parameters
        ----------
        filename : str
            Checkpoint filename relative to the experiment directory.
        """

        self._save_model(
            filename,
            {
                "encoder_state": self.encoder.state_dict(),
                "decoder_state": self.decoder.state_dict(),
                "model_optimizer_state": self.model_optimizer.state_dict(),
                "dataset": getattr(self.args, "rank", 0),
            },
        )


__all__ = ["DecoderFinetuner"]
