"""Audio preprocessing utilities for StarNet-style stem datasets."""

import datetime
import os
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from tqdm import tqdm

from music_star.utils import copy_files, setup_logger

logger = setup_logger(__name__, "log_wav.log")


@dataclass
class Audio:
    """Preprocess raw stem WAV files into split HDF5 datasets.

    Parameters
    ----------
    data_path : pathlib.Path
        Source directory containing ``TRACK.STEM.wav`` files.
    preprocessed_path : pathlib.Path
        Working directory for copied, trimmed, and silence-filtered WAV files.
    h5_path : pathlib.Path
        Output directory for HDF5 files.
    split_path : pathlib.Path
        Output directory for train/validation/test splits.
    stems : list
        Stem ids to process.
    in_sr : int
        Input sample rate.
    out_sr : int
        Output sample rate.
    in_channel : int
        Input channel count.
    out_channel : int
        Output channel count.
    file_type : str
        Audio file extension.
    train_ratio : float
        Training split ratio.
    val_ratio : float
        Validation split ratio.
    """

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
        """Create preprocessing output directories."""

        os.makedirs(self.preprocessed_path, exist_ok=True)
        os.makedirs(self.h5_path, exist_ok=True)
        os.makedirs(self.split_path, exist_ok=True)

    @staticmethod
    def file_length(file_path: Path):
        """Return the duration of a WAV file.

        Parameters
        ----------
        file_path : pathlib.Path
            WAV file to inspect with ``ffprobe``.

        Returns
        -------
        float
            Duration in seconds. Empty or unreadable outputs return ``0``.
        """

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
            capture_output=True,
        ).stdout
        try:
            duration = float(output)
        except ValueError:
            duration = 0
            logger.error(f"The file from {file_path} is empty")
        return duration

    def copy_to_preprocessed(self):
        """Copy raw input files into the preprocessing workspace."""

        files = [f for f in self.data_path.glob("**/*")]
        copy_files(files, self.data_path, self.preprocessed_path)

    def stems_duration(self, data_path: Path, track_name: str):
        """Return durations for all configured stems of one track.

        Parameters
        ----------
        data_path : pathlib.Path
            Directory containing WAV files.
        track_name : str
            Track id prefix.

        Returns
        -------
        list[float]
            Duration for each configured stem.
        """

        durations = []
        for i in self.stems:
            file = Path(data_path) / f"{track_name}.{i}.wav"
            file_len = self.file_length(file)
            # duration = float("{:.1f}".format(file_len))
            durations.append(file_len)
        return durations

    def stems_duration_compare(self, data_path: Path):
        """Find tracks whose stems have unequal durations.

        Parameters
        ----------
        data_path : pathlib.Path
            Directory containing WAV files.

        Returns
        -------
        dict[str, list[float]]
            Mapping from track id to stem durations for tracks needing trim.
        """

        to_trim = {}
        for track_name in self.track_names(data_path):
            durations = self.stems_duration(data_path, track_name)
            if len(set(durations)) > 1:
                to_trim[track_name] = durations
        return to_trim

    @staticmethod
    def track_names(data_path: Path):
        """Return sorted track ids from ``TRACK.STEM.wav`` filenames.

        Parameters
        ----------
        data_path : pathlib.Path
            Directory containing WAV files.

        Returns
        -------
        list[str]
            Sorted unique track ids.
        """

        return sorted({f.name.split(".")[0] for f in Path(data_path).glob("*.wav")})

    def audio_trim(self):
        """Trim stem files so each track has equal stem durations."""

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
                    capture_output=True,
                )
            logger.info(f"Track No. {name} files are trimmed after {int_min_length} seconds")

    def stems_silence_detect(self, data_path: Path, track_name: str, duration: int = 1):
        """Detect silent intervals for each stem in a track.

        Parameters
        ----------
        data_path : pathlib.Path
            Directory containing WAV files.
        track_name : str
            Track id prefix.
        duration : int, optional
            Minimum silence duration in seconds.

        Returns
        -------
        tuple[list, list, list, list]
            Silence starts, ends, durations, and total silent time per stem.
        """

        s_starts = []
        s_durations = []
        s_ends = []
        s_total = []

        for i in self.stems:
            file = str(Path(data_path) / f"{track_name}.{i}.wav")
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
            total = 0.0

            for _, item in enumerate(stamps):
                fields = item.split("=")
                if "start" in fields[0]:
                    starts.append(float(fields[1]))
                elif "duration" in fields[0]:
                    durations.append(float(fields[1]))
                    total += float(fields[1])
                elif "end" in fields[0]:
                    ends.append(float(fields[1]))

            s_starts.append(starts)
            s_ends.append(ends)
            s_durations.append(durations)
            s_total.append(total)

        return s_starts, s_ends, s_durations, s_total

    def silence_remove(self):
        """Remove intervals where the most silent stem is silent.

        The same intervals are removed from all stems so the files remain
        aligned. Calling the method repeatedly can remove additional silence
        dominated by other stems.
        """
        from pydub import AudioSegment

        data_path = self.preprocessed_path

        for track_name in tqdm(self.track_names(data_path), desc="silence removal"):
            lengths = self.stems_duration(data_path, track_name)
            starts, ends, durations, total = self.stems_silence_detect(data_path, track_name, 1)

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
                    output_file = output_file + wav[s_ends[e] * 1000 : s_starts[e + 1] * 1000]

                output_file = output_file + wav[s_ends[-1] * 1000 : lengths[0] * 1000]

                output_file.export(f"{self.preprocessed_path}/{file}", "wav")

            logger.info(
                "Track No. %s after silence removal: File lengths: %s",
                track_name,
                self.stems_duration(self.preprocessed_path, track_name),
            )

    def total_duration(self, data_path: Path):
        """Return total duration for a preprocessed domain.

        Parameters
        ----------
        data_path : pathlib.Path
            Directory containing aligned WAV files.

        Returns
        -------
        datetime.timedelta
            Sum of the first stem duration for every track.
        """

        duration = 0
        for track_name in self.track_names(data_path):
            duration += self.stems_duration(data_path, track_name)[0]
        return datetime.timedelta(seconds=duration)

    def save_as_h5(self):
        """Convert preprocessed WAV files into HDF5 files."""

        import h5py
        from scipy.io import wavfile

        data_path = self.preprocessed_path

        for f in tqdm(sorted(os.listdir(data_path)), desc="converting to hdf5"):
            input_file_path = Path(data_path) / f
            output_file_path = self.h5_path / input_file_path.with_suffix(".h5").name
            with NamedTemporaryFile(suffix=".wav") as output_wav_file:
                subprocess.run(
                    [
                        "ffmpeg",
                        "-v",
                        "quiet",
                        "-y",
                        "-i",
                        str(input_file_path),
                        "-f",
                        "wav",
                        "-ar",
                        str(self.out_sr),
                        "-ac",
                        str(self.out_channel),
                        output_wav_file.name,
                    ],
                    capture_output=True,
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
        """Filter files by stem id.

        Parameters
        ----------
        all_files : Iterable[pathlib.Path]
            Candidate files.
        stem_no : int
            Stem id to keep.

        Returns
        -------
        list[pathlib.Path]
            Files matching the stem id.
        """

        return [f for f in all_files if (f.is_file() and f.name[4] == str(stem_no))]

    def split(self):
        """Split HDF5 stem files into train, validation, and test directories."""

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
        """Run the full preprocessing pipeline."""

        self.copy_to_preprocessed()
        self.audio_trim()
        self.silence_remove()
        self.save_as_h5()
        self.split()


class PitchAugmentation:
    """Randomly pitch-shift part of a waveform segment.

    Parameters
    ----------
    args
        Namespace containing ``aug_mag`` and ``sample_rate``.
    """

    def __init__(self, args):
        self.magnitude = args.aug_mag
        self.sample_rate = args.sample_rate

    def __call__(self, wav):
        """Apply local pitch shift augmentation.

        Parameters
        ----------
        wav : numpy.ndarray
            Input waveform segment.

        Returns
        -------
        numpy.ndarray
            Augmented waveform segment.
        """

        length = wav.shape[0]
        if length < 4:
            return wav

        shift_length = random.randint(max(length // 4, 1), max(length // 2, 1))
        shift_start = random.randint(0, max(length - shift_length, 0))
        shift_end = shift_start + shift_length
        shift_pitch = (np.random.rand() - 0.5) * 2 * self.magnitude
        import librosa

        shifted = librosa.effects.pitch_shift(
            wav[shift_start:shift_end],
            sr=self.sample_rate,
            n_steps=shift_pitch,
        )
        return np.concatenate([wav[:shift_start], shifted, wav[shift_end:]])
