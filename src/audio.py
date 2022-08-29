import datetime
import os
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

import h5py
import numpy as np
from pydub import AudioSegment
from scipy.io import wavfile
from tqdm import tqdm

import utils
from utils import copy_files

logger = utils.setup_logger(__name__, "log_wav.log")


@dataclass
class Audio:
    data_path: Path
    preprocessed_path: Path
    h5_path: Path
    split_path: Path
    stems: list
    in_sr: int
    out_sr: int
    in_channel: int
    out_channel: int
    file_type: str
    train_ratio: float
    val_ratio: float

    def __post_init__(self):
        os.makedirs(self.preprocessed_path, exist_ok=True)
        os.makedirs(self.h5_path, exist_ok=True)
        os.makedirs(self.split_path, exist_ok=True)

    @staticmethod
    def file_length(file_path: Path):
        """returns the length of the wav file"""
        output = subprocess.run(
            [
                "ffprobe",
                "-show_entries",
                "format=duration",
                "-v",
                "quiet",
                "-print_format",
                "compact=print_section=0:nokey=1:escape=csv",
                str(file_path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout
        try:
            duration = float(output)
        except ValueError:
            duration = 0
            logger.error(f"The file from {file_path} is empty")
        return duration

    def copy_to_preprocessed(self):
        files = [f for f in self.data_path.glob("**/*")]
        copy_files(files, self.data_path, self.preprocessed_path)

    def stems_duration(self, data_path: Path, track_name: str):
        """returns the length of all stems of a track"""
        durations = []
        for i in self.stems:
            file = f"{str(data_path)}/{track_name}.{i}.wav"
            file_len = self.file_length(file)
            # duration = float("{:.1f}".format(file_len))
            durations.append(file_len)
        return durations

    def stems_duration_compare(self, data_path: Path):
        """returns the tracks that should be trimmed for equal length"""
        to_trim = {}
        prev_track = ""
        for f in sorted(os.listdir(data_path)):
            track_name = f[0:3]
            if prev_track != track_name:
                durations = self.stems_duration(data_path, track_name)
                if len(set(durations)) > 1:
                    to_trim[track_name] = durations
            prev_track = track_name
        return to_trim

    def audio_trim(self):
        """trims the end of audio stems to achieve equal length"""
        to_trim = self.stems_duration_compare(self.preprocessed_path)
        if not to_trim:
            logger.info("No file needs trimming")
            return
        for name, ps in tqdm(to_trim.items(), desc="trimming"):
            min_length = min(ps)
            max(ps)
            ps.index(min(ps))
            int_min_length = int(min_length) - 1
            for i in self.stems:
                file = f"{name}.{i}.wav"
                subprocess.run(
                    [
                        "ffmpeg",
                        "-i",
                        f"{self.data_path}/{file}",
                        "-y",
                        "-ss",
                        str(0),
                        "-to",
                        str(int_min_length),
                        "-c",
                        "copy",
                        "-f",
                        "wav",
                        "-ar",
                        str(self.in_sr),
                        "-ac",
                        str(self.in_channel),
                        f"{self.preprocessed_path}/{file}",
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
            logger.info(
                f"Track No. {name} files are trimmed after {int_min_length} seconds"
            )

    def stems_silence_detect(self, data_path: Path, track_name: str, duration: int = 1):
        """return the silent timestamps of a track's stems"""
        s_starts = []
        s_durations = []
        s_ends = []
        s_total = []

        for i in self.stems:
            file = f"{str(data_path)}/{track_name}.{i}.wav"
            output = subprocess.Popen(
                [
                    "ffmpeg",
                    "-i",
                    file,
                    "-af",
                    f"silencedetect=n=-40dB:d={duration},ametadata=print:file=-",
                    "-f",
                    "null",
                    "-",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            grep = subprocess.Popen(
                ["grep", "-E", "start|end|duration"],
                stdin=output.stdout,
                stdout=subprocess.PIPE,
                encoding="utf-8",
            )
            output.stdout.close()
            stamps = grep.communicate()[0].splitlines()

            starts = []
            durations = []
            ends = []
            total = 0

            for _, item in enumerate(stamps):
                item = item.split("=")
                if "start" in item[0]:
                    starts.append(float(item[1]))
                elif "duration" in item[0]:
                    durations.append(float(item[1]))
                    total += float(item[1])
                elif "end" in item[0]:
                    ends.append(float(item[1]))

            s_starts.append(starts)
            s_ends.append(ends)
            s_durations.append(durations)
            s_total.append(total)

        return s_starts, s_ends, s_durations, s_total

    def silence_remove(self):
        """removes the silence of the most silent instrument and synch all the stems.
        can call it multiple times to remove the silence of the second most silent instruments
        """
        data_path = self.preprocessed_path

        for f in tqdm(sorted(os.listdir(data_path)), desc="silence removal"):
            track_name = f[0:3]
            lengths = self.stems_duration(data_path, track_name)
            starts, ends, durations, total = self.stems_silence_detect(
                data_path, track_name, 1
            )

            for p in self.stems:
                if len(starts[p]) == len(ends[p]) + 1:
                    ends[p].append(lengths[p])
                    total[p] = total[p] + lengths[p] - starts[p][-1]

            most_silence = max(total)
            most_silent = total.index(most_silence)

            s_starts = starts[most_silent]
            s_ends = ends[most_silent]

            if len(s_starts) == 0:
                continue

            for p in self.stems:
                file = f"{track_name}.{p}.wav"
                wav = AudioSegment.from_wav(f"{data_path}/{file}")

                output_file = wav[: s_starts[0] * 1000]

                for e in range(len(s_ends) - 1):
                    output_file = (
                        output_file + wav[s_ends[e] * 1000 : s_starts[e + 1] * 1000]
                    )

                output_file = output_file + wav[s_ends[-1] * 1000 : lengths[0] * 1000]

                output_file.export(f"{self.preprocessed_path}/{file}", "wav")

        logger.info(
            f"Track No. {track_name} after silence removal:"
            f"File lengths: {self.stems_duration(self.preprocessed_path, track_name)}"
        )

    def total_duration(self, data_path: Path):
        """returns the total duration of the specified domain"""
        duration = 0
        for f in sorted(os.listdir(data_path)):
            track_name = f[0:3]
            duration += self.stems_duration(data_path, track_name)[0]
        return datetime.timedelta(seconds=duration)

    def save_as_h5(self):
        """saving wav data as h5 files"""
        data_path = self.preprocessed_path

        for f in tqdm(sorted(os.listdir(data_path)), desc="converting to hdf5"):
            output_file_path = self.h5_path / f"{f.strip(self.file_type)}h5"
            with NamedTemporaryFile(suffix=".wav") as output_wav_file:
                subprocess.run(
                    [
                        "ffmpeg",
                        "-v",
                        "quiet",
                        "-y",
                        "-i",
                        f"{data_path}/{f}",
                        "-f",
                        "wav",
                        "-ar",
                        str(self.out_sr),
                        "-ac",
                        str(self.out_channel),
                        output_wav_file.name,
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )

                rate, wav_data = wavfile.read(output_wav_file.name)
                assert wav_data.dtype == np.int16
                wav = wav_data.astype("float")

                with h5py.File(output_file_path, "w") as output_file:
                    chunk_shape = (min(10000, len(wav)),)
                    wav_dataset = output_file.create_dataset(
                        "wav", wav.shape, dtype=wav.dtype, chunks=chunk_shape
                    )
                    wav_dataset[...] = wav

        logger.info("Converted files to HDF5")

    @staticmethod
    def filter_stems(all_files, stem_no):
        """filters the file of specific type (wav)"""
        return [f for f in all_files if (f.is_file() and f.name[4] == str(stem_no))]

    def split(self):
        """split instrument tracks separately into train, val, and test set for every instrument"""
        stem_names = ["cv", "clarinet", "vibra", "sp", "strings", "piano"]
        for sn in stem_names:
            os.makedirs(self.split_path / sn, exist_ok=True)

        input_files = sorted(os.listdir(self.h5_path))
        pieces = list(range(len(input_files) // len(self.stems)))

        n_train = int(len(pieces) * self.train_ratio)
        n_val = int(len(pieces) * self.val_ratio)
        assert n_val > 0
        n_test = len(pieces) - n_train - n_val

        random.shuffle(pieces)

        for s in self.stems:
            stem_files = self.filter_stems(self.h5_path.glob("**/*"), s)
            dst = Path(self.split_path / stem_names[s])
            train_files = [stem_files[i] for i in pieces[:n_train]]
            val_files = [stem_files[i] for i in pieces[n_train : n_train + n_val]]
            test_files = [stem_files[i] for i in pieces[n_train + n_val :]]
            copy_files(train_files, self.h5_path, dst / "train")
            copy_files(val_files, self.h5_path, dst / "val")
            copy_files(test_files, self.h5_path, dst / "test")
        logger.info(
            "Data splitted as follows: Train - %s, Validation - %s, Test - %s",
            n_train,
            n_val,
            n_test,
        )

    def preprocess_audio(self):
        self.copy_to_preprocessed()
        self.audio_trim()
        self.silence_remove()
        self.save_as_h5()
        self.split()
