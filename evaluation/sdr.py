import os
import sys
import pandas
import librosa
import numpy as np
import pretty_midi
from tqdm import tqdm
import essentia.standard as estd
from mir_eval.separation import bss_eval_sources


SR = 16000

def SDR(output, reference):

    length = output.shape[0]
    sdr, a, b, c = bss_eval_sources(reference[:length], output[:length])
    return float(sdr)
 

def main():
    top = 'samples'
    ref_dir = 'reduced'
    dirs = ['single','pipeline', 'unsup','config1', 'config3']
    domains =  {'cv':0, 'ps':3}
    c1 = [d + '.cv' for d in dirs]
    c2 = [d + '.ps' for d in dirs]
    cols = c1 + c2
    df = pandas.DataFrame(columns=cols)
    
    for i, d  in tqdm(enumerate(dirs)):
        for dom in domains.keys():
            where = f'{top}/{d}/{dom}'
            for j, f in enumerate(sorted(os.listdir(where))):
                name = f[0:3]
                output = f'{where}/{f}'
                reference = f'{top}/{ref_dir}/{name}.{domains.get(dom)}.wav'
                outa, _ = librosa.load(output, SR)
                refa, _ = librosa.load(reference, SR)
                sdr = SDR(outa, refa)
                df.at[j, f'{d}.{dom}'] = sdr       
    df.to_csv('./sdr.csv')


if __name__ == "__main__":
    main()