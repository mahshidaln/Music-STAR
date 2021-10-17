import os
import sys
import shutil
from pathlib import Path


def main():
    data_dir = './data/star'
    for f in sorted(os.listdir(Path(data_dir))):
        track_name = f[0:3]
        os.makedirs(Path(f'{data_dir}/{track_name}'), exist_ok = True)
        source = f[4]
        if(source == '0'):
            shutil.move(f'{data_dir}/{f}', f'{data_dir}/{track_name}/mixture.wav')
        elif(source == '1'):
            shutil.move(f'{data_dir}/{f}', f'{data_dir}/{track_name}/clarinet.wav')
        elif(source == '2'):
            shutil.move(f'{data_dir}/{f}', f'{data_dir}/{track_name}/vibra.wav')
        elif(source == '3'):
            shutil.move(f'{data_dir}/{f}', f'{data_dir}/{track_name}/counter.wav')
        elif(source == '4'):
            shutil.move(f'{data_dir}/{f}', f'{data_dir}/{track_name}/string.wav')
        elif(source == '5'):
            shutil.move(f'{data_dir}/{f}', f'{data_dir}/{track_name}/piano.wav')

if __name__ == "__main__":
    main()