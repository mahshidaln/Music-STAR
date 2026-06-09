"""Music-STAR stem-supervised trainer."""

from __future__ import annotations

from itertools import chain
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from music_star.models import LegacyEncoder, WaveNet
from music_star.models.universal import cross_entropy_loss
from music_star.trainers.base import (
    BaseTrainer,
    data_paths,
    epoch_length,
    epochs,
    paired_loaders,
    to_device,
)
from music_star.utils import LossManager


class MusicStarStemTrainer(BaseTrainer):
    """Train the Music-STAR stem-supervised two-decoder solution.

    Parameters
    ----------
    args
        Namespace with source mixture paths, two target stem paths, model fields,
        and optimizer settings.
    """

    checkpoint_keys = ("encoder_state", "decoder_state", "decoder2_state")

    def __init__(self, args: Any):
        super().__init__(args)
        self.args.n_datasets = len(data_paths(args))
        self.data = [
            (
                paired_loaders(args, path, Path(args.target)),
                paired_loaders(args, path, Path(args.target2)),
            )
            for path in data_paths(args)
        ]
        self.train_loss = LossManager("stem supervised")
        self.eval_loss = LossManager("eval stem supervised")

        self.encoder = LegacyEncoder(args).to(self.device)
        self.decoder = WaveNet(args).to(self.device)
        self.decoder2 = WaveNet(args).to(self.device)
        self.model_optimizer = optim.Adam(
            chain(self.encoder.parameters(), self.decoder.parameters(), self.decoder2.parameters()),
            lr=float(args.lr),
        )
        self.lr_manager = optim.lr_scheduler.ExponentialLR(
            self.model_optimizer, float(getattr(args, "lr_decay", 0.99))
        )

    def train_batch(
        self,
        source_mix: torch.Tensor,
        target_a: torch.Tensor,
        target_b: torch.Tensor,
    ) -> float:
        """Train one stem-supervised Music-STAR batch.

        Parameters
        ----------
        source_mix : torch.Tensor
            Source mixture encoded by the trainable encoder.
        target_a : torch.Tensor
            First target stem for teacher forcing and reconstruction.
        target_b : torch.Tensor
            Second target stem for teacher forcing and reconstruction.

        Returns
        -------
        float
            Sum of the two decoder reconstruction losses.
        """

        source_mix = to_device(source_mix.float(), self.device)
        target_a = to_device(target_a.float(), self.device)
        target_b = to_device(target_b.float(), self.device)
        code = self.encoder(source_mix)
        loss_a = cross_entropy_loss(self.decoder(target_a, code), target_a).mean()
        loss_b = cross_entropy_loss(self.decoder2(target_b, code), target_b).mean()
        loss = loss_a + loss_b

        self.model_optimizer.zero_grad()
        loss.backward()
        self._clip(
            chain(self.encoder.parameters(), self.decoder.parameters(), self.decoder2.parameters())
        )
        self.model_optimizer.step()

        self.train_loss.add(float(loss.detach().cpu()))
        return float(loss.detach().cpu())

    def train(self) -> None:
        """Run stem-supervised Music-STAR training and checkpointing."""

        best_eval = float("inf")
        for epoch in range(self.start_epoch, self.start_epoch + epochs(self.args)):
            self.encoder.train()
            self.decoder.train()
            self.decoder2.train()
            self.train_loss.reset()

            with tqdm(total=epoch_length(self.args), desc=f"Train epoch {epoch}") as train_enum:
                for batch_num in range(epoch_length(self.args)):
                    dataset_index = batch_num % self.args.n_datasets
                    (train_a, _), (train_b, _) = self.data[dataset_index]
                    source_mix, target_a = next(train_a)
                    _, target_b = next(train_b)
                    batch_loss = self.train_batch(source_mix, target_a, target_b)
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
        """Evaluate the stem-supervised Music-STAR model.

        Parameters
        ----------
        epoch : int
            Epoch number used for progress display.

        Returns
        -------
        float
            Mean validation loss summed across both decoders.
        """

        self.encoder.eval()
        self.decoder.eval()
        self.decoder2.eval()
        self.eval_loss.reset()
        total_batches = max(int(np.ceil(epoch_length(self.args) / 10)), 1)
        with tqdm(total=total_batches, desc=f"Eval epoch {epoch}") as valid_enum, torch.no_grad():
            for batch_num in range(total_batches):
                dataset_index = batch_num % self.args.n_datasets
                (_, valid_a), (_, valid_b) = self.data[dataset_index]
                source_mix, target_a = next(valid_a)
                _, target_b = next(valid_b)
                source_mix = to_device(source_mix.float(), self.device)
                target_a = to_device(target_a.float(), self.device)
                target_b = to_device(target_b.float(), self.device)
                code = self.encoder(source_mix)
                loss = cross_entropy_loss(self.decoder(target_a, code), target_a).mean()
                loss = loss + cross_entropy_loss(self.decoder2(target_b, code), target_b).mean()
                self.eval_loss.add(float(loss.detach().cpu()))
                valid_enum.update()
        return self.eval_loss.epoch_mean()

    def save_model(self, filename: str) -> None:
        """Save stem-supervised encoder, both decoders, and optimizer.

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
                "decoder2_state": self.decoder2.state_dict(),
                "model_optimizer_state": self.model_optimizer.state_dict(),
                "dataset": getattr(self.args, "rank", 0),
            },
        )


__all__ = ["MusicStarStemTrainer"]
