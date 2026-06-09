"""Timbre-style evaluation with a triplet similarity network."""

import os

import librosa
import pandas
from tqdm.auto import tqdm

from music_star.evaluation import triplet_network

triplet_model, triplet_backbone = triplet_network.build_model(num_features=12)
triplet_model.load_weights("checkpoint.ckpt")


SR = 16000

MFCC_KWARGS = {"n_mfcc": 13, "hop_length": 500}


def timbre_eval(output, reference, neg):
    """Compute triplet-network timbre similarity.

    Parameters
    ----------
    output : numpy.ndarray
        Generated audio waveform.
    reference : numpy.ndarray
        Positive reference waveform.
    neg : numpy.ndarray
        Negative reference waveform.

    Returns
    -------
    float
        Cosine similarity score for output/reference timbre.
    """

    mfcc_out = librosa.feature.mfcc(output, sr=SR, **MFCC_KWARGS)[1:]
    mfcc_ref = librosa.feature.mfcc(reference, sr=SR, **MFCC_KWARGS)[1:]
    mfcc_neg = librosa.feature.mfcc(neg, sr=SR, **MFCC_KWARGS)[1:]

    mfcc_triplet_cos, _ = triplet_model.predict(
        [(mfcc_out.T[None, :, :], mfcc_ref.T[None, :, :], mfcc_neg.T[None, :, :])]
    ).reshape(2)
    return mfcc_triplet_cos


def main():
    """Run the timbre evaluation script and write ``timbre.csv``."""

    top = "results"
    ref_dir = "refs"
    dirs = ["single", "pipeline", "unsup", "config1", "config3"]
    domains = {"cv": 0, "ps": 3}
    c1 = [d + ".cv" for d in dirs]
    cols = c1
    df = pandas.DataFrame(columns=cols)
    dom = "ps"
    for d in tqdm(dirs):
        where = f"{top}/{d}/{dom}"
        for j, f in enumerate(sorted(os.listdir(where))):
            if f[-3:] != "wav":
                j = j - 1
                continue
            name = f[0:3]
            output = f"{where}/{f}"
            pos = f"{top}/{ref_dir}/{name}.{domains.get(dom)}.wav"
            neg = f"{top}/{ref_dir}/{name}.0.wav"
            outa, _ = librosa.load(output, 16000)
            posa, _ = librosa.load(pos, 16000)
            nega, _ = librosa.load(neg, 16000)
            cos = timbre_eval(outa, posa, nega)
            df.at[j, f"{d}.{dom}"] = cos
    df.to_csv("./timbre.csv")


if __name__ == "__main__":
    main()
