import os
import sys
import torch
from torch import distributed

from train import AutoencoderTrainer, MusicStarTrainer
from parser import get_name, get_parser
from audio import WavFilesDataset
from data import Dataset, Splitter


def main():
    parser = get_parser()
    args = parser.parse_args()
    print(args)
    sys.exit()
    if "preprocess" in args.operation_mode:
        print("prep")
        dataset = WavFilesDataset(args.input)
        dataset.save_as_hd5(args.h5_path)
        splitter = Splitter(args)
        #TODO
    
    if "train" in args.operation_mode:
        print("train")

        if args.world_size > 1:
            torch.cuda.set_device(args.rank % torch.cuda.device_count())
            distributed.init_process_group(backend=args.dist_backend,
                                       init_method="tcp://" + args.master,
                                       rank=args.rank,
                                       world_size=args.world_size)

        if(args.train_step == 1):
            print("1")
            AutoencoderTrainer(args).train()     
        else:
            print("2")
            MusicStarTrainer(args).train()

    if "translate" in args.operation_mode:
        print("translate")
        #TODO

    if "analyze" in args.operation_mode:
        print("analyze")  
        #TODO      

if __name__ == '__main__':
    main()
