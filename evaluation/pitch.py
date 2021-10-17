import os
import sys
import pandas
import librosa
import numpy as np
import pretty_midi
from tqdm import tqdm
import essentia.standard as estd


SR = 16000

MFCC_KWARGS = dict(
    n_mfcc=13,
    hop_length=500
)


def get_pitches(audio):
    pitches = estd.MultiPitchMelodia(sampleRate=SR)(audio)
    pitches = [[pretty_midi.utilities.hz_to_note_number(p) for p in pl if not np.isclose(0, p)]
               for pl in pitches]    
    pitches = [[int(p + 0.5) for p in pl] for pl in pitches]
    return pitches

def pitch_jaccard(output, reference):
    pitches_output, pitches_reference = get_pitches(output), get_pitches(reference)
    assert len(pitches_output) == len(pitches_reference)
    jaccard = []
    for pl_output, pl_reference in zip(pitches_output, pitches_reference):
        matches = len(set(pl_output) & set(pl_reference))
        total = len(set(pl_output) | set(pl_reference))
        if total == 0:
            jaccard.append(0)
        else:
            jaccard.append(1 - matches / total)
    jaccard = np.mean(jaccard)
    return jaccard


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
                jaccard = pitch_jaccard(outa, refa)
                df.at[j, f'{d}.{dom}'] = jaccard       
    df.to_csv('./lsd.csv')
                

if __name__ == "__main__":
    main()