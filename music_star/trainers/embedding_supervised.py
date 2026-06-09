"""Embedding-supervised Music-STAR trainer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from music_star.checkpoints import is_legacy_encoder_state, load_checkpoint_state, load_saved_args
from music_star.data import StarDataset
from music_star.models import Encoder, LegacyEncoder
from music_star.trainers.base import BaseTrainer, data_paths, epoch_length, epochs, to_device
from music_star.utils import LossManager


class EmbeddingSupervisedTrainer(BaseTrainer):
    """Train a mixture encoder to mimic a frozen source encoder.

    Parameters
    ----------
    args
        Namespace with paired data paths, pretrained checkpoint, and optimizer
        settings.
    """

    checkpoint_keys = ("encoder_state",)

    def __init__(self, args: Any):
        super().__init__(args)
        self.args.n_datasets = len(data_paths(args))
        mix_path = Path(args.mix)
        self.data = [StarDataset(args, path, mix_path) for path in data_paths(args)]
        self.star_loss = LossManager("star loss")
        self.eval_loss = LossManager("eval loss")

        pretrained_dir = Path(args.pretrained_checkpoint).parent
        model_args = load_saved_args(pretrained_dir)
        checkpoint = load_checkpoint_state(pretrained_dir, Path(args.pretrained_checkpoint).name)

        encoder_class = (
            LegacyEncoder if is_legacy_encoder_state(checkpoint["encoder_state"]) else Encoder
        )
        self.encoder = encoder_class(model_args).to(self.device)
        self.encoder.load_state_dict(checkpoint["encoder_state"], strict=True)
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

        self.star_encoder = LegacyEncoder(args).to(self.device)
        self.model_optimizer = optim.Adam(self.star_encoder.parameters(), lr=float(args.lr))
        self.lr_manager = optim.lr_scheduler.ExponentialLR(
            self.model_optimizer, float(getattr(args, "lr_decay", 0.98))
        )

    def train_batch(self, x: torch.Tensor, mix: torch.Tensor) -> float:
        """Train one embedding-supervised batch.

        Parameters
        ----------
        x : torch.Tensor
            Target stem audio batch for the frozen universal encoder.
        mix : torch.Tensor
            Mixture audio batch for the trainable encoder.

        Returns
        -------
        float
            L1 latent matching loss.
        """

        x = to_device(x.float(), self.device)
        mix = to_device(mix.float(), self.device)
        with torch.no_grad():
            target_code = self.encoder(x)
        mix_code = self.star_encoder(mix)
        loss = F.l1_loss(mix_code, target_code)

        self.model_optimizer.zero_grad()
        loss.backward()
        self._clip(self.star_encoder.parameters())
        self.model_optimizer.step()

        self.star_loss.add(float(loss.detach().cpu()))
        return float(loss.detach().cpu())

    def train(self) -> None:
        """Run embedding-supervised encoder training and checkpointing."""

        best_eval = float("inf")
        for epoch in range(self.start_epoch, self.start_epoch + epochs(self.args)):
            self.star_encoder.train()
            self.encoder.eval()
            self.star_loss.reset()

            with tqdm(total=epoch_length(self.args), desc=f"Train epoch {epoch}") as train_enum:
                for batch_num in range(epoch_length(self.args)):
                    dataset_index = batch_num % self.args.n_datasets
                    x, mix = next(self.data[dataset_index].train_iter)
                    batch_loss = self.train_batch(x, mix)
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
        """Evaluate the embedding-supervised encoder.

        Parameters
        ----------
        epoch : int
            Epoch number used for progress display.

        Returns
        -------
        float
            Mean validation L1 loss.
        """

        self.encoder.eval()
        self.star_encoder.eval()
        self.eval_loss.reset()
        total_batches = max(int(np.ceil(epoch_length(self.args) / 10)), 1)
        with tqdm(total=total_batches, desc=f"Eval epoch {epoch}") as valid_enum, torch.no_grad():
            for batch_num in range(total_batches):
                dataset_index = batch_num % self.args.n_datasets
                x, mix = next(self.data[dataset_index].valid_iter)
                x = to_device(x.float(), self.device)
                mix = to_device(mix.float(), self.device)
                loss = F.l1_loss(self.star_encoder(mix), self.encoder(x))
                self.eval_loss.add(float(loss.detach().cpu()))
                valid_enum.update()
        return self.eval_loss.epoch_mean()

    def save_model(self, filename: str) -> None:
        """Save the embedding-supervised encoder and optimizer state.

        Parameters
        ----------
        filename : str
            Checkpoint filename relative to the experiment directory.
        """

        self._save_model(
            filename,
            {
                "encoder_state": self.star_encoder.state_dict(),
                "model_optimizer_state": self.model_optimizer.state_dict(),
                "dataset": getattr(self.args, "rank", 0),
            },
        )


StarLatentTrainer = EmbeddingSupervisedTrainer

__all__ = ["EmbeddingSupervisedTrainer", "StarLatentTrainer"]
