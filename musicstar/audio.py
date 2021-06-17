import os
import sys
import h5py
import tqdm
import torch
import random
import librosa
import datetime
import shutil
import subprocess
import numpy as np
import torch.utils.data as data

from pathlib import Path
from scipy.io import wavfile
from pydub import AudioSegment
from tempfile import NamedTemporaryFile

import utils
from utils import mu_law, copy_files

logger = utils.setup_logger(__name__, 'log_wav.log')


class WavFilesDataset(data.Dataset):
    """Process the wav audio files"""
   
    def __init__(self, args):
        self.path = args.data_path
        self.epoch_length = args.epoch_length #10000 seconds
        self.segment_length = args.segment_length #16000

        self.input_rate = args.input_rate #44100
        self.sample_rate = args.sample_rate #16000
        
        self.in_channel = args.in_channel # 2
        self.out_channel = args.out_channel # 1

        self.file_types = [args.file_type] if args.file_type else ['wav']
        self.file_paths = self.filter_paths(self.path.glob('**/*'), self.file_types)
        
        self.fft_size = args.fft_size
        self.windows_length = args.windows_length
        self.hop_size = args.hop_size


    @staticmethod
    def filter_paths(all_files, file_types):
        """filters the file of specific type (wav)"""
        return [f for f in all_files
                if (f.is_file()
                    and any(f.name.endswith(suffix) for suffix in file_types)
                    and '__MACOSX' not in f.parts)]

    @staticmethod
    def filter_stems(all_files, stem_no):
        """filters the file of specific type (wav)"""
        return [f for f in all_files
                if (f.is_file() and f.name[-5]==str(stem_no))]

    @staticmethod
    def file_length(file_path):
        """returns the length of the wav file"""
        output = subprocess.run(['ffprobe',
                                 '-show_entries', 'format=duration',
                                 '-v', 'quiet',
                                 '-print_format', 'compact=print_section=0:nokey=1:escape=csv',
                                 str(file_path)],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE).stdout
        duration = float(output)
        return duration


    def parts_duration(self, file_path, file_name):
        """returns the length of all parts of a track"""
        durations = []
        for i in range(0, self.parts):
            file = f'{str(file_path)}/{file_name}.{i}.wav'
            file_len = self.file_length(file)
            duration = float("{:.1f}".format(file_len))
            durations.append(duration)
        return durations


    def parts_duration_compare(self, data_path, log=False):
        """returns the tracks that should be trimmed for equal length"""
        to_trim = {}
        prev_track = ''
        logger.info('Parts Duration')
        for f in sorted(os.listdir(data_path)):        
            track_name = f[0:3]
            if(prev_track != track_name):
                durations = self.parts_duration(data_path, track_name, self.parts)
                if (len(set(durations)) > 1):
                    if (log):
                        logger.info(f'Track No. {track_name}. ' f'File lengths: {durations}')
                    to_trim[track_name] = durations
            prev_track = track_name
        return to_trim


    def audio_trim(self, data_path, replace=True):
        """trims the end of audio parts to achieve equal length"""
        to_trim = self.parts_duration_compare(data_path, self.parts, False)
        for name, ps in tqdm.tqdm(to_trim.items()):
            min_length = min(ps)
            max_length = max(ps)
            min_arg = ps.index(min(ps))
            for i in range(self.parts):
                file = f'{name}.{i}.wav'
                if(replace):
                    with NamedTemporaryFile() as output_file:
                        subprocess.run(['ffmpeg',
                        '-i', f'{data_path}/{file}',
                        '-y',
                        '-ss', str(0),
                        '-to', str(min_length),
                        '-c', 'copy',
                        '-f', 'wav',
                        '-ar', str(self.input_rate), 
                        '-ac', str(self.in_channel), 
                        output_file.name
                        ],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE)

                        shutil.copyfile(output_file.name, f'{data_path}/{file}')
                else:  
                    output = Path(data_path.parent / 'trim')
                    output.mkdir(parents=True, exist_ok=True)                      
                    subprocess.run(['ffmpeg',
                        '-i', f'{data_path}/{file}',
                        '-ss', str(0),
                        '-to', str(min_length),
                        '-c', 'copy',
                        f'{output}/{file}'
                        ])  
    

    def parts_silence_detect(self, track_path, track_name, duration=1):
        """return the silent timestamps of a track's parts"""
        s_starts = []
        s_durations = []
        s_ends = []
        s_total = []

        for i in range(0, self.parts):
            file = f'{str(track_path)}/{track_name}.{i}.wav'
            output = subprocess.Popen(['ffmpeg',
                                '-i', file,
                                '-af', f'silencedetect=n=-40dB:d={duration},ametadata=print:file=-',
                                '-f', 'null',
                                '-',
                                ],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE
                                )                                     
            grep = subprocess.Popen(['grep','-E', 'start|end|duration' ], 
                                    stdin=output.stdout, stdout=subprocess.PIPE, encoding='utf-8')                      
            output.stdout.close()
            stamps = grep.communicate()[0].splitlines()

            starts = []
            durations = []
            ends = []
            total = 0

            for _, item in enumerate(stamps):
                item = item.split('=')
                if('start' in item[0]):
                    starts.append(float(item[1]))
                elif('duration' in item[0]):
                    durations.append(float(item[1]))
                    total += float(item[1])
                elif('end' in item[0]):
                    ends.append(float(item[1]))
                

            s_starts.append(starts)
            s_ends.append(ends)
            s_durations.append(durations)
            s_total.append(total)   

        return s_starts, s_ends, s_durations, s_total
 

    def silence_remove(self, data_path, replace=True):
        """removes the silence and synch all the parts"""
        logger.info('parts duration after removing silence')
        for f in tqdm.tqdm(sorted(os.listdir(data_path))):        
            track_name = f[0:3]
            durations = self.parts_duration(data_path, track_name, self.parts)
            starts, ends, durations, total = self.parts_silence_detect(data_path, track_name, self.parts, 1)
            
            most_silence = max(total)
            most_silent = total.index(most_silence)  

            s_starts = starts[most_silent]
            s_ends = ends[most_silent]      

            if(len(s_starts) == 0):
                continue
            elif(len(s_starts) == len(s_ends)+1):
                s_ends.append(durations[0])
                #s_durations.append(durations[0]) 
            for p in range(self.parts):
                file = f'{track_name}.{p}.wav'
                wav = AudioSegment.from_wav(f'{data_path}/{file}')
                if(replace):
                    output_file = wav[:s_starts[0]*1000]
                    for e in range(len(s_ends)-1):
                        output_file = output_file + wav[s_ends[e]*1000:s_starts[e+1]*1000]
                    output_file = output_file + wav[s_ends[len(s_ends)-1]*1000:durations[0]*1000]
                    output_file.export(f'{data_path}/{file}', 'wav')
                else:  
                    output = Path(data_path.parent / 'no_silence')
                    output.mkdir(parents=True, exist_ok=True)                      
                    #TODO 
            logger.info(f'Track No. {track_name}. ' f'File lengths: {self.parts_duration(data_path, track_name, self.parts)}')       


    def total_duration(self, data_path):
        """returns the total duration of the specified domain"""
        duration = 0
        for f in sorted(os.listdir(data_path)):  
            track_name = f[0:3]
            duration += self.parts_duration(data_path, track_name)[0]
        return datetime.timedelta(seconds=duration/3)


    def _random_file(self):
        """picks a track randomly and return the specific instrument file"""
        #track_no = f'{np.random.randint(len(self.file_paths)//args.stems):03}'
        #track_name = f'{track_no}.{part}.wav'
        return random.choice(self.file_paths)


    def file_segment(self, data_path, start_time):
        """returns a segment file starting at start_time with duration of segment_length"""
        length_sec = self.segment_length / self.sample_rate
        length_sec += .01  # just in case
        with NamedTemporaryFile() as output_file:
            output = subprocess.run(['ffmpeg',
                                     '-v', 'quiet',
                                     '-y',  # overwrite
                                     '-ss', str(start_time),
                                     '-i', str(data_path),
                                     '-t', str(length_sec),
                                     '-f', 'wav',
                                     '-ar', str(self.sample_rate),
                                     '-ac', self.out_channel, 
                                     output_file.name
                                     ],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE).stdout
            rate, wav_data = wavfile.read(output_file)
            assert wav_data.dtype == np.int16
            wav = wav_data[:self.segment_len].astype('float')

            return wav


    def random_file_segment(self):
        """picks a file randomly and segment a random slice"""
        file = self._random_file()
        file_length_sec = self.file_length(file)
        segment_length_sec = self.segment_length / self.sample_rate
        start_time = random.random() * (file_length_sec - segment_length_sec * 2)  # just in case
        wav_data = self.file_segment(file, start_time)
        if len(wav_data) != self.segment_length:
            logger.warn('File "%s" has length %s, segment length is %s, wav data length: %s',
                        file, file_length_sec, segment_length_sec, len(wav_data))

        return file, file_length_sec, start_time, wav_data


    def get_random_slice(self):
        """returns the wav data of the selected slice"""
        wav_data = None
        while wav_data is None or len(wav_data) != self.segment_length:         
            file, file_length_sec, start_time, wav_data = self.random_file_segment()
        logger.debug('Sample: File: %s, File length: %s, Start time: %s',
                    file, file_length_sec, start_time)

        return wav_data


    def __len__(self):
        """get number of data samples"""
        return self.epoch_length 


    def __getitem__(self):
        """returns the tensor of the segmented data from the specified part of a track"""
        wav = self.get_random_slice()
        return torch.FloatTensor(wav)

   
    def save_as_h5(self, output: Path):
        """saving wav data as h5 files"""
        for file_path in tqdm.tqdm(self.file_paths):
            output_file_path = output / file_path.relative_to(self.path).with_suffix('.h5')
            output_file_path.parent.mkdir(parents=True, exist_ok=True)

            with NamedTemporaryFile(suffix='.wav') as output_wav_file:
                logger.debug(f'Converting {file_path} to {output_wav_file.name}')
                subprocess.run(['ffmpeg',
                                '-v', 'quiet',
                                '-y', 
                                '-i', file_path,
                                '-f', 'wav',
                                '-ar', str(self.sample_rate), 
                                '-ac', str(self.out_channel),  
                                output_wav_file.name
                                ],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
                
                rate, wav_data = wavfile.read(output_wav_file.name)
                assert wav_data.dtype == np.int16
                wav = wav_data.astype('float')

                with h5py.File(output_file_path, 'w') as output_file:
                    chunk_shape = (min(10000, len(wav)),)
                    wav_dataset = output_file.create_dataset('wav', wav.shape, dtype=wav.dtype,
                                                          chunks=chunk_shape)
                    wav_dataset[...] = wav

                logger.debug(f'Saved input {file_path} to {output_file_path}. '
                             f'Wav length: {wav.shape}')

class Splitter:
    """split instrument tracks separately into train, val, and test set"""
    def __init__(self, args, input_path):
        self.output_path = Path(args.split_path)
        input_files = WavFilesDataset.filter_paths(input_path.glob('**/*'), ['wav'])
        n_train = int(len(input_files)//args.stems * args.train_ratio)
        n_val = int(len(input_files)//args.stems * args.val_ratio)
        if n_val == 0:
            n_val = 1
        n_test = len(input_files) - n_train - n_val
        assert n_test > 0
        stems = []
        for s in range(args.stems):
            dst = Path(self.output_path / f'{input_path.name}_{s}')
            dst.mkdir(exist_ok=True, parents=True)
            stem_files = WavFilesDataset.filter_stems(input_files, s)
            random.shuffle(stem_files)
            copy_files(stem_files[:n_train], input_path, dst / 'train')
            copy_files(stem_files[n_train:n_train + n_val], input_path, dst / 'val')
            copy_files(stem_files[n_train + n_val:], input_path, dst / 'test')
            logger.info('Split stem %s of path %s as follows: Train - %s, Validation - %s, Test - %s', s, input_path.name, n_train, n_val, n_test)      


class PitchAugmentation:
    """Randomly shift the pitch of 0.25 to 0.5 seconds"""
    def __init__(self, args):
        self.magnitude = args.aug_mag
        self.sample_rate = args.sample_rate

    def __call__(self, wav):
        length = wav.shape[0]
        shift_length = random.randint(length // 4, length // 2) #segment of length between 0.25 and 0.5 seconds
        shift_start = random.randint(0, length // 2)
        shift_end = shift_start + shift_length
        shift_pitch = (np.random.rand() - 0.5) * 2 * self.magnitude

        aug_wav = np.concatenate([wav[:shift_start],
                              librosa.effects.pitch_shift(wav[shift_start:shift_end], self.sample_rate, shift_pitch), 
                              wav[shift_end:]])

        return aug_wav


class SignAugmentation:
    #TODO
    def __init__(self, args):
        self.sample_rate = args.sample_rate

    def __call__(self, wav):
        aug_wav = wav
        return aug_wav


class RemixAugmentation:
    #TODO
    def __init__(self, args):
        self.sample_rate = args.sample_rate

    def __call__(self, wav):
        aug_wav = wav
        return aug_wav


