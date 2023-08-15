'''
# Post processing module for data recording
# Author: Kevin Geng @Laronix, Sep. 2022

# Load log.csv, generate standard wav files with selected samplerate, and calculate stastitical features
'''

from random import sample
import librosa
import soundfile as sf
import numpy as np
import pdb
from pathlib import Path
import sys
import pandas as pd
indir = Path(sys.argv[1])
assert indir.exists() == True 
wavs = Path(indir/Path("Audio_to_Evaluate")).glob("**/*.wav")
log = Path(indir/Path("log.csv"))

# x = np.loadtxt(log, dtype=str, delimiter=",")
x = pd.read_csv(log, header=0)

# y, sr = librosa.load("/home/kevingeng/laronix_automos/Julianna/Audio_to_evaluate/tmp0kgcdpi2.wav", sr=48000)
outdir = indir/Path("output")
# pdb.set_trace()
# outdir_clean = indir/Path("output_clean")
Path.mkdir(outdir, exist_ok=True)
# Path.mkdir(outdir_clean, exist_ok=True)
## Capitalize E valuate
# for i, j in zip(x["Audio_to_Evaluate"], x["Reference_ID"]):
#     y, sr = librosa.load(i, sr=48000)
#     # kevin 1017 John's trial with original data.
#     y_ = librosa.util.normalize(y, norm=5)
#     y_cut, index = librosa.effects.trim(y_, top_db=30)
#     # normalized and cut
#     # pdb.set_trace()
#     # sf.write(outdir/Path(str(indir)+"_"+ j +".wav"), y_cut, samplerate=sr)
#     sf.write(outdir/Path(Path(indir).stem+"_"+ j +".wav"), y_cut, samplerate=sr)

def process_audio(file_path, ref_id, sr=48000, norm=5, top_db=30):
    y, _ = librosa.load(file_path, sr=sr)
    y_norm = librosa.util.normalize(y, norm=norm)
    y_cut, _ = librosa.effects.trim(y_norm, top_db=top_db)
    return y_cut

def save_audio(y_cut, ref_id, outdir, indir, sr=48000):
    out_path = outdir / f"{Path(indir).stem}_{ref_id}.wav"
    sf.write(out_path, y_cut, samplerate=sr)

def main(audio_files, ref_ids, outdir, indir):
    for file_path, ref_id in zip(audio_files, ref_ids):
        y_cut = process_audio(file_path, ref_id)
        save_audio(y_cut, ref_id, outdir, indir)

main(x["Audio_to_Evaluate"], x["Reference_ID"], outdir, indir)
