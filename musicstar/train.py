import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
import torch.distributed
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_value_

from data import Dataset, Universe_Dataset
from wavenet import WaveNet
from model_universe import Encoder
from model_musicstar import MusicStarEncoder
from utils import train_logger

torch.backends.cudnn.benchmark = True
torch.multiprocessing.set_start_method('spawn', force=True)


class UniverseTrainer:
    """training the universe-based model for multi-instrument translation"""
    def __init__(self, args):
        self.args = args
        self.data = Universe_Dataset(args)

        self.expPath = args.checkpoint / 'Universe' / args.exp_name
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

        #pretrained model
        pretrained = args.pretrained.parent.glob(args.checkpoint.name + '_0.pth')

        #pretrained args
        pretrained_args = torch.load(args.pretrained.parent / 'args.pth')[0]

        #pretrained encoder
        self.pretrained_encoder = Encoder(pretrained_args)
        self.pretrained_encoder.load_state_dict(torch.load(pretrained)['encoder_state'])
        self.pretrained_encoder = torch.nn.DataParallel(self.pretrained_encoder).cuda()

        for param in self.pretrained_encoder.parameters():
            param.requires_grad = False
        
        #universe encoder
        self.universe_encoder = Encoder(args)
        if args.world_size > 1:
            self.universe_encoder = torch.nn.parallel.DistributedDataParallel(self.universe_encoder)
        else:                                                                    
            self.universe_encoder = torch.nn.DataParallel(self.universe_encoder).cuda()
     

        #losses
        self.train_loss = LossManager('eval loss') 
        self.eval_loss = LossManager('train loss')

        #optimizers
        self.optimizer = optim.Adam(universe_encoder.parameters(), lr=args.lr)    

        #learning rates
        self.lr_manager = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, args.lr_decay)
        self.lr_manager.last_epoch = self.start_epoch

        #resume training
        if args.resume:
            checkpoint_args_file = self.expPath / 'args.pth'
            checkpoint_args = torch.load(checkpoint_args_file)

            last_epoch = checkpoint_args[-1]
            self.start_epoch = last_epoch + 1

            checkpoint_state_file = self.expPath / f'lastmodel_{last_epoch}.pth'
            states = torch.load(args.checkpoint_state_file)

            self.universe_encoder.load_state_dict(states['encoder_state'])

            if(args.load_optimizer):
                self.optimizer.load_state_dict(states['optimizer_state'])
        else:
            self.start_epoch = 0

    def train_batch(self, x, mix):
        x_code = self.pretrianed_encoder(x)
        mix_code = self.universe_encoder(mix)
        
        train_loss = F.l1_loss(mix_code, x_code)
        #train_loss = F.MSE(y, z)
        self.optimizer.zero_grad()
        train_loss.backward()

        if self.args.grad_clip is not None:
                clip_grad_value_(self.universe_encoder.parameters(), self.args.grad_clip)

        self.optimizer.step()

        self.train_loss.add(train_loss.data.item())
        return train_loss.data.item()
    
    def train_epoch(self, epoch):
        self.train_loss.reset()

        self.universe_encoder.train()
        self.pretrained_encoder.eval()

        total_batches = self.args.epoch_length

        with tqdm(total=total_batches, desc='Train epoch %d' % epoch) as train_enum:
            for batch_num in range(total_batches):

                x, mix = next(self.data.train_iter)
                x = x.to(self.device)
                mix = mix.to(self.device)
                x = x.float()
                mix = mix.float()

                batch_loss = self.train_batch(x, mix)

                train_enum.set_description(f'Train (loss: {batch_loss:.2f}) epoch {epoch}')
                train_enum.update()

    def eval_batch(self, x, mix):
        z = self.pretrained_encoder(x)
        y = self.universe_encoder(mix)
        
        eval_loss = F.l1_loss(y, z)
        self.eval_loss.add(eval_loss.data.item())
        return eval_loss.data.item()

    def evaluate_epoch(self, epoch): 
        self.eval_loss.reset()

        self.pretrained_encoder.eval()
        self.universe_encoder.eval()

        total_batches = int(np.ceil(self.args.epoch_len / 10))

        with tqdm(total=total_batches) as valid_enum, torch.no_grad():
            for batch_num in range(total_batches):

                x, mix = next(self.data.valid_iter)
                x = x.to(self.device)
                mix = mix.to(self.device)
                x = x.float()
                mix = mix.float()
                batch_loss = self.eval_batch(x, mix)

                valid_enum.set_description(f'Test (loss: {batch_loss:.2f}) epoch {epoch}')
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

            if self.args.save_epoch:
                self.save_model(f'lastmodel_{epoch}_rank_{self.args.rank}.pth')
            else:
                self.save_model(f'lastmodel_{self.args.rank}.pth')

            self.lr_manager.step()
            self.logger.debug('Ended epoch')                 

    def save_model(self, filename):
        save_to = self.expPath / filename
        torch.save({'encoder_state': self.universe_encoder.module.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'dataset': self.args.data,
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