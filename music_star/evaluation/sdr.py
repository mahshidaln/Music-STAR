"""Signal-to-distortion ratio evaluation for generated audio."""

import os

import librosa
import pandas
from mir_eval.separation import bss_eval_sources
from tqdm import tqdm

SR = 16000


def SDR(output, reference):
    """Compute SDR between output and reference signals.

    Parameters
    ----------
    output : numpy.ndarray
        Generated audio waveform.
    reference : numpy.ndarray
        Reference audio waveform.

    Returns
    -------
    float
        Signal-to-distortion ratio.
    """

    length = output.shape[0]
    sdr, _, _, _ = bss_eval_sources(reference[:length], output[:length])
    return float(sdr)


def main():
    """Run the SDR evaluation script and write ``sdr.csv``."""

    top = "samples"
    ref_dir = "reduced"
    dirs = ["single", "pipeline", "unsup", "config1", "config3"]
    domains = {"cv": 0, "ps": 3}
    c1 = [d + ".cv" for d in dirs]
    c2 = [d + ".ps" for d in dirs]
    cols = c1 + c2
    df = pandas.DataFrame(columns=cols)

    for d in tqdm(dirs):
        for dom in domains:
            where = f"{top}/{d}/{dom}"
            for j, f in enumerate(sorted(os.listdir(where))):
                name = f[0:3]
                output = f"{where}/{f}"
                reference = f"{top}/{ref_dir}/{name}.{domains.get(dom)}.wav"
                outa, _ = librosa.load(output, SR)
                refa, _ = librosa.load(reference, SR)
                sdr = SDR(outa, refa)
                df.at[j, f"{d}.{dom}"] = sdr
    df.to_csv("./sdr.csv")


if __name__ == "__main__":
    main()
