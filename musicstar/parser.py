import os
import argparse
from pathlib import Path


def get_parser():
    parser = argparse.ArgumentParser()
    
    # Main options
    parser.add_argument('--operation-mode', type=str, default=["preprocess", "train", "translate", "analyze"], required=True, nargs='+',
                        help='What is the operation mode? "preprocess, "train", "translate", "analyze"?')
    parser.add_argument('--train-step', type=int, default=1,
                        help='step 1: single enc-dec training, step 2: music-star encoder training') 
    
    # Training options:
    #parser.add_argument('--gpu', type=str, required=True, help='Specify which GPUs to use separated by a comma. Ex: 2,3')
    parser.add_argument('--epochs', type=int, default=10000, metavar='N',
                        help='number of epochs to train (default: 92)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--exp-name', type=str, required=True,
                        help='Experiment name')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training')                        
    parser.add_argument('--checkpoint', type=Path, 
                        help='Directory of chechpoints path')                        
    parser.add_argument('--load-optimizer', action='store_true')
    parser.add_argument('--save-model', action='store_true',
                        help='Save model per epoch')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--lr-decay', type=float, default=0.98,
                        help='new LR = old LR * decay')                        
                    

    # Data options
    parser.add_argument('--data-path',
                        type=Path, help='Data path', nargs='+')
    parser.add_argument('--split-path',
                        type=Path, help='path to train, validation, and test wav files')                           
    parser.add_argument('--h5-path',
                        type=Path, help='HD5 data path of splitted files')                                                
    parser.add_argument('--segment-length', type=int, default=16000,
                        help='Segment length in samples')
    parser.add_argument('--input-rate', type=int, default=44100,
                        help='sample rate of input files')                        
    parser.add_argument('--sample-rate', type=int, default=44100,
                        help='sample rate of output files') 
    parser.add_argument('--file-type', type=str, default='wav', 
                        help='Encoded File type')          
    parser.add_argument('--in-channel', type=int, default=2,
                        help='number of input audio channels')                         
    parser.add_argument('--out-channel', type=int, default=1,
                        help='number of output audio channels')               
    parser.add_argument('--stems', type=int, default=3,
                        help='number of stems including the mixture') 
    parser.add_argument('--train-ratio', type=int, default=80,
                        help='ratio of data used for training')                          
    parser.add_argument('--val-ratio', type=int, default=15,
                        help='ratio of data used for validation')  
    parser.add_argument('--epoch-length', type=int, default=1000,
                        help='number of 1-second segments per epoch') 
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--num-workers', type=int, default=10,
                        help='DataLoader workers')
    parser.add_argument('--data-aug', action='store_true',
                        help='Turns data aug on')
    parser.add_argument('--aug-mag', type=float, default=0.5,
                        help='Data augmentation magnitude.')
    

     # Distributed options
    parser.add_argument("--rank", default=0, type=int)
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--master")
    parser.add_argument('--dist-backend', default='nccl')
    
    
    parser.add_argument('--local_rank', type=int,
                        help='Ignored during training.')


    #STFT options
    parser.add_argument('--fft-size', type=int, default=2048,
                        help='Size of FFT')
    parser.add_argument('--epoch-len', type=int, default=2047,
                        help='Size of the windoe')                        
    parser.add_argument('--hop-size', type=int, default=80,
                        help='Samples per epoch')                        

    
    # Encoder1 options
    parser.add_argument('--latent-d', type=int, default=128,
                        help='Latent size')
    parser.add_argument('--repeat-num', type=int, default=6,
                        help='No. of hidden layers')
    parser.add_argument('--encoder-channels', type=int, default=128,
                        help='Hidden layer size')
    parser.add_argument('--encoder-blocks', type=int, default=3,
                        help='No. of encoder blocks.')
    parser.add_argument('--encoder-pool', type=int, default=800,
                        help='Number of encoder outputs to pool over.')
    parser.add_argument('--encoder-final-kernel-size', type=int, default=1,
                        help='final conv kernel size')
    parser.add_argument('--encoder-layers', type=int, default=10,
                        help='No. of layers in each encoder block.')
    parser.add_argument('--encoder-func', type=str, default='relu',
                        help='Encoder activation func.')

    # Decoder options
    parser.add_argument('--blocks', type=int, default=4,
                        help='No. of wavenet blocks.')
    parser.add_argument('--layers', type=int, default=10,
                        help='No. of layers in each block.')
    parser.add_argument('--kernel-size', type=int, default=2,
                        help='Size of kernel.')
    parser.add_argument('--residual-channels', type=int, default=128,
                        help='Residual channels to use.')
    parser.add_argument('--skip-channels', type=int, default=128,
                        help='Skip channels to use.')

    # Discriminator options
    parser.add_argument('--d-layers', type=int, default=3,
                        help='Number of 1d 1-kernel convolutions on the input Z vectors')
    parser.add_argument('--d-channels', type=int, default=100,
                        help='1d convolutions channels')
    parser.add_argument('--d-cond', type=int, default=1024,
                        help='WaveNet conditioning dimension')
    parser.add_argument('--d-weight', type=float, default=1e-2,
                        help='Adversarial loss weight.')
    parser.add_argument('--p-dropout-discriminator', type=float, default=0.0,
                        help='Discriminator input dropout - if unspecified, no dropout applied')
    parser.add_argument('--grad-clip', type=float,
                        help='If specified, clip gradients with specified magnitude')  
    """
    # MusicSTAR Encoder options
    parser.add_argument('--latent-d', type=int, default=128,
                        help='Latent size')
    parser.add_argument('--repeat-num', type=int, default=6,
                        help='No. of hidden layers')
    parser.add_argument('--encoder-channels', type=int, default=128,
                        help='Hidden layer size')
    parser.add_argument('--encoder-blocks', type=int, default=3,
                        help='No. of encoder blocks.')
    parser.add_argument('--encoder-pool', type=int, default=800,
                        help='Number of encoder outputs to pool over.')
    parser.add_argument('--encoder-final-kernel-size', type=int, default=1,
                        help='final conv kernel size')
    parser.add_argument('--encoder-layers', type=int, default=10,
                        help='No. of layers in each encoder block.')
    parser.add_argument('--encoder-func', type=str, default='relu',
                        help='Encoder activation func.')
    """                                              
                            

