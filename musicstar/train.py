import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from itertools import chain
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.utils import clip_grad_value_


torch.backends.cudnn.benchmark = True
torch.multiprocessing.set_start_method('spawn', force=True)

from data import Dataset
from wavenet import WaveNet
from model_universe import cross_entropy_loss, Encoder, ZDiscriminator
from model_musicstar import MusicStarEncoder
from utils import train_logger


class AutoencoderTrainer:
    """training the autoencoder for the first step of training"""
    def __init__(self, args):
        self.args = args
        self.data = [Dataset(args, domain_path) for domain_path in args.data]
        self.expPath = args.checkpoint / 'Autoencoder' / args.exp_name
        if not self.expPath.exists():
             self.expPath.mkdir(parents=True, exist_ok=True)
        self.logger = train_logger(self.args, self.expPath)
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        #seed
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        
        #modules
        self.encoder = Encoder(args)
        self.decoder = WaveNet(args)
        self.discriminator = ZDiscriminator(args)

        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)
        self.discriminator = self.discriminator.to(self.device)

        #distributed
        if args.world_size > 1:
            self.encoder = torch.nn.parallel.DistributedDataParallel(self.encoder,
                                                                    device_ids=[torch.cuda.current_device()],
                                                                    output_device=torch.cuda.current_device())
            self.discriminator = torch.nn.parallel.DistributedDataParallel(self.discriminator,
                                                                    device_ids=[torch.cuda.current_device()],
                                                                    output_device=torch.cuda.current_device())
            self.decoder = torch.nn.parallel.DistributedDataParallel(self.decoder,
                                                                    device_ids=[torch.cuda.current_device()],
                                                                    output_device=torch.cuda.current_device())

        #losses
        self.reconstruction_loss = [LossManager(f'train reconstruction {i}') for i in range(len(self.data))]
        self.discriminator_loss = LossManager('train discriminator')
        self.total_loss = LossManager('train total')

        self.reconstruction_val = [LossManager(f'validation reconstruction {i}') for i in range(len(self.data))]
        self.discriminator_val = LossManager('validation discriminator')
        self.total_val  = LossManager('validation total')

        #optimizers
        self.autoenc_optimizer = optim.Adam(chain(self.encoder.parameters(), self.decoder.parameters()), 
                                            lr=args.lr)
        self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(), 
                                            lr=args.lr)

        #resume training
        if args.resume:
            checkpoint_args_file = self.expPath / 'args.pth'
            checkpoint_args = torch.load(checkpoint_args_file)

            last_epoch = checkpoint_args[-1]
            self.start_epoch = last_epoch + 1

            checkpoint_state_file = self.expPath / f'lastmodel_{last_epoch}.pth'
            states = torch.load(args.checkpoint_state_file)

            self.encoder.load_state_dict(states['encoder_state'])
            self.decoder.load_state_dict(states['decoder_state'])
            self.discriminator.load_state_dict(states['discriminator_state'])

            if(args.load_optimizer):
                self.autoenc_optimizer.load_state_dict(states['autoenc_optimizer_state'])
                self.discriminator_optimizer.load_state_dict(states['discriminator_optimizer_state'])
            self.logger.info('Loaded checkpoint parameters')
        else:
            self.start_epoch = 0


        #learning rates
        self.lr_manager = torch.optim.lr_scheduler.ExponentialLR(self.model_optimizer, args.lr_decay)
        self.lr_manager.last_epoch = self.start_epoch
        self.lr_manager.step()
       

    def train_epoch(self, epoch):
        #modules
        self.encoder.train()
        self.decoder.train()
        self.discriminator.train()


        #losses
        for lm in self.reconstruction_loss:
            lm.reset()
        self.discriminator_loss.reset()
        self.total_loss.reset()

        total_batches = self.args.epoch_length // self.args.batch_size

        with tqdm(total=total_batches, desc=f'Train epoch {epoch}') as train_enum:
            for batch_num in range(total_batches):
                if self.args.world_size > 1:
                    dataset_no = self.args.rank
                else:
                    dataset_no = batch_num % self.args.n_datasets

                x, x_aug = next(self.data[dataset_no].train_iter)

                x = x.to(self.device)
                x_aug = x_aug.to(self.device)
               
                x, x_aug = x.float(), x_aug.float()

                # Train discriminator
                z = self.encoder(x)
                z_logits = self.discriminator(z)
                discriminator_loss = F.cross_entropy(z_logits, torch.tensor([dataset_no] * x.size(0)).long().cuda()).mean()
                loss = discriminator_loss * self.args.d_weight
                self.discriminator_optimizer.zero_grad()
                loss.backward()
                if self.args.grad_clip is not None:
                    clip_grad_value_(self.discriminator.parameters(), self.args.grad_clip)           
                self.discriminator_optimizer.step()

                # Train autoencoder
                z = self.encoder(x_aug)
                y = self.decoder(x, z)
                z_logits = self.discriminator(z)
                discriminator_loss = - F.cross_entropy(z_logits, torch.tensor([dataset_no] * x.size(0)).long().cuda()).mean()
                reconstruction_loss = cross_entropy_loss(y, x)
                self.reconstruction_loss[dataset_no].add(reconstruction_loss.data.cpu().numpy().mean())
                loss = (reconstruction_loss.mean() + self.args.d_weight * discriminator_loss)
                self.model_optimizer.zero_grad()
                loss.backward()

                if self.args.grad_clip is not None:
                    clip_grad_value_(self.encoder.parameters(), self.args.grad_clip)
                    clip_grad_value_(self.decoder.parameters(), self.args.grad_clip)
                self.model_optimizer.step()

                self.loss_total.add(loss.data.item())

                train_enum.set_description(f'Train (loss: {loss.data.item():.2f}) epoch {epoch}')
                train_enum.update()

    def validate_epoch(self, epoch):
        
        #modules
        self.encoder.eval()
        self.decoder.eval()
        self.discriminator.eval()

        #losses
        for lm in self.reconstruction_val:
            lm.reset()
        self.discriminator_val.reset()
        self.total_val.reset()

        total_batches = self.args.epoch_length // self.args.batch_size // 10

        with tqdm(total=total_batches) as valid_enum, torch.no_grad():
            for batch_num in range(total_batches):
                
                if self.args.world_size > 1:
                    dataset_no = self.args.rank
                else:
                    dataset_no = batch_num % self.args.n_datasets

                x, x_aug = next(self.data[dataset_no].valid_iter)

                x = x.to(self.device)
                x_aug = x.to(self.device)
                
                x, x_aug = x.float(), x_aug.float()

                z = self.encoder(x)
                y = self.decoder(x, z)
                z_logits = self.discriminator(z)
                z_classification = torch.max(z_logits, dim=1)[1]
                z_accuracy = (z_classification == dataset_no).float().mean()

                self.discriminator_val.add(z_accuracy.data.item())

                # discriminator_right = F.cross_entropy(z_logits, dset_num).mean()
                discriminator_right = F.cross_entropy(z_logits, torch.tensor([dataset_no] * x.size(0)).long().cuda()).mean()
                recon_loss = cross_entropy_loss(y, x)

                self.evals_recon[dataset_no].add(recon_loss.data.cpu().numpy().mean())

                total_loss = discriminator_right.data.item() * self.args.d_lambda + recon_loss.mean().data.item()

                self.total_val.add(total_loss)

                valid_enum.set_description(f'Test (loss: {total_loss:.2f}) epoch {epoch}')
                valid_enum.update()
    
    def train(self):
        best_loss = float('inf')
        for epoch in range(self.start_epoch, self.args.epochs):
            self.logger.info(f'Starting epoch, Rank {self.args.rank}, Dataset: {self.args.data[self.args.rank]}')
            self.train_epoch(epoch)
            self.validate_epoch(epoch)

            train_losses = [self.reconstruction_loss, self.discriminator_loss]
            val_losses = [self.reconstruction_val, self.discriminator_val]

            self.logger.info(f'Epoch %s Rank {self.args.rank} - Train loss: (%s), Validation loss (%s)',
                             epoch, train_losses, val_losses)

            mean_loss = self.val_total.epoch_mean()
            if mean_loss < best_loss:
                self.save_model(f'bestmodel_{self.args.rank}.pth')
                best_loss = mean_loss

            if self.args.save_model:
                self.save_model(f'lastmodel_{epoch}_rank_{self.args.rank}.pth')
            else:
                self.save_model(f'lastmodel_{self.args.rank}.pth')

           # if self.args.rank:
            #    torch.save([self.args, epoch], '%s/args.pth' % self.expPath)
                   
            self.lr_manager.step()
            self.logger.debug('Ended epoch')                 

    def save_model(self, filename):
        save_to = self.expPath / filename
        torch.save({'encoder_state': self.encoder.module.state_dict(),
                    'decoder_state': self.decoder.module.state_dict(),
                    'discriminator_state': self.discriminator.module.state_dict(),
                    'autoenc_optimizer_state': self.autoenc_optimizer.state_dict(),
                    'd_optimizer_state': self.discriminator_optimizer.state_dict(),
                    'dataset': self.args.rank,
                    }, save_to)

        self.logger.debug(f'Saved model to {save_to}')


class MusicStarTrainer:
    """training the musicSTAR encoder for the second step of training"""
    def __init__(self, args):
        #TODO
        self.args = args
        #self.data = [Dataset(args, domain_path) for domain_path in args.data]
        self.expPath = args.checkpoint / 'MusicStar' / args.exp_name
        self.logger = train_logger(self.args, self.expPath)

        self.encoder = MusicStarEncoder(args)
        self.decoder = WaveNet(args)

class LossManager():
    """manage the list of losses"""
    def __init__(self, name):
        self.name = name
        self.losses = []

    def reset(self):
        self.losses = []

    def add(self, val):
        self.losses.append(val)

    def epoch_mean(self):
        if self.losses:
            return np.mean(self.losses)
        else:
            return 0

    def losses_sum(self):
        return sum(self.losses)