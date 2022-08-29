import random
from parser import get_parser

from audio import Audio


def main():
    parser = get_parser()
    args = parser.parse_args()
    random.seed(args.seed)

    audio = Audio(
        data_path=args.data_path,
        preprocessed_path=args.preprocessed_path,
        h5_path=args.h5_path,
        split_path=args.split_path,
        stems=[i for i in range(args.stems)],
        in_sr=args.input_rate,
        out_sr=args.sample_rate,
        in_channel=args.in_channel,
        out_channel=args.out_channel,
        file_type=args.file_type if args.file_type else "wav",
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    audio.preprocess_audio()


if __name__ == "__main__":
    main()
