"""Adversarial universal translation trainer."""

from __future__ import annotations

from itertools import chain
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from music_star.data import Dataset
from music_star.models import Encoder, WaveNet, ZDiscriminator
from music_star.models.universal import cross_entropy_loss
from music_star.trainers.base import BaseTrainer, data_paths, epoch_length, epochs, to_device
from music_star.utils import LossManager


class AdversarialUniversalTrainer(BaseTrainer):
    """Train the Universal Music Translation adversarial baseline.

    Parameters
    ----------
    args
        Namespace with data, model, optimizer, and discriminator settings.
    """

    checkpoint_keys = ("encoder_state", "decoder_state", "discriminator_state")

    def __init__(self, args: Any):
        super().__init__(args)
        self.args.n_datasets = len(data_paths(args))
        self.data = [Dataset(args, path) for path in data_paths(args)]

        self.recon_losses = [LossManager(f"recon {i}") for i in range(self.args.n_datasets)]
        self.discriminator_loss = LossManager("discriminator")
        self.total_loss = LossManager("total")
        self.eval_total = LossManager("eval total")

        self.encoder = Encoder(args).to(self.device)
        self.decoder = WaveNet(args).to(self.device)
        self.discriminator = ZDiscriminator(args).to(self.device)

        self.model_optimizer = optim.Adam(
            chain(self.encoder.parameters(), self.decoder.parameters()), lr=float(args.lr)
        )
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=float(args.lr))
        self.lr_manager = optim.lr_scheduler.ExponentialLR(
            self.model_optimizer, float(getattr(args, "lr_decay", 0.98))
        )

    def train_batch(self, x: torch.Tensor, x_aug: torch.Tensor, dataset_index: int) -> float:
        """Train one adversarial universal batch.

        Parameters
        ----------
        x : torch.Tensor
            Original mu-law audio batch.
        x_aug : torch.Tensor
            Augmented audio batch used for encoder training.
        dataset_index : int
            Domain index for discriminator labels.

        Returns
        -------
        float
            Scalar generator loss for the batch.
        """

        x = to_device(x.float(), self.device)
        x_aug = to_device(x_aug.float(), self.device)
        target = torch.full((x.size(0),), dataset_index, dtype=torch.long, device=self.device)

        z = self.encoder(x)
        z_logits = self.discriminator(z)
        discriminator_right = F.cross_entropy(z_logits, target).mean()
        d_loss = discriminator_right * float(self.args.d_lambda)

        self.d_optimizer.zero_grad()
        d_loss.backward()
        self._clip(self.discriminator.parameters())
        self.d_optimizer.step()
        self.discriminator_loss.add(float(d_loss.detach().cpu()))

        z = self.encoder(x_aug)
        y = self.decoder(x, z)
        z_logits = self.discriminator(z)
        discriminator_wrong = -F.cross_entropy(z_logits, target).mean()
        recon_loss = cross_entropy_loss(y, x)
        loss = recon_loss.mean() + float(self.args.d_lambda) * discriminator_wrong

        self.model_optimizer.zero_grad()
        loss.backward()
        self._clip(chain(self.encoder.parameters(), self.decoder.parameters()))
        self.model_optimizer.step()

        self.recon_losses[dataset_index].add(float(recon_loss.detach().mean().cpu()))
        self.total_loss.add(float(loss.detach().cpu()))
        return float(loss.detach().cpu())

    def train(self) -> None:
        """Run adversarial universal training and checkpointing."""

        best_eval = float("inf")
        for epoch in range(self.start_epoch, epochs(self.args)):
            self.encoder.train()
            self.decoder.train()
            self.discriminator.train()
            for meter in [*self.recon_losses, self.discriminator_loss, self.total_loss]:
                meter.reset()

            with tqdm(total=epoch_length(self.args), desc=f"Train epoch {epoch}") as train_enum:
                for batch_num in range(epoch_length(self.args)):
                    dataset_index = batch_num % self.args.n_datasets
                    x, x_aug = next(self.data[dataset_index].train_iter)
                    batch_loss = self.train_batch(x, x_aug, dataset_index)
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
        """Evaluate the universal model for one epoch.

        Parameters
        ----------
        epoch : int
            Epoch number used for progress display.

        Returns
        -------
        float
            Mean validation loss.
        """

        self.encoder.eval()
        self.decoder.eval()
        self.discriminator.eval()
        self.eval_total.reset()

        total_batches = max(int(np.ceil(epoch_length(self.args) / 10)), 1)
        with tqdm(total=total_batches, desc=f"Eval epoch {epoch}") as valid_enum, torch.no_grad():
            for batch_num in range(total_batches):
                dataset_index = batch_num % self.args.n_datasets
                x, _ = next(self.data[dataset_index].valid_iter)
                x = to_device(x.float(), self.device)
                target = torch.full(
                    (x.size(0),), dataset_index, dtype=torch.long, device=self.device
                )
                z = self.encoder(x)
                y = self.decoder(x, z)
                discriminator_right = F.cross_entropy(self.discriminator(z), target).mean()
                recon_loss = cross_entropy_loss(y, x)
                loss = recon_loss.mean() + float(self.args.d_lambda) * discriminator_right
                self.eval_total.add(float(loss.detach().cpu()))
                valid_enum.update()
        return self.eval_total.epoch_mean()

    def save_model(self, filename: str) -> None:
        """Save universal encoder, decoder, discriminator, and optimizers.

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
                "discriminator_state": self.discriminator.state_dict(),
                "model_optimizer_state": self.model_optimizer.state_dict(),
                "d_optimizer_state": self.d_optimizer.state_dict(),
                "dataset": getattr(self.args, "rank", 0),
            },
        )


__all__ = ["AdversarialUniversalTrainer"]
