# Kevin @ Laronix Dec. 2022
# Data processing at Laronix
import csv
import soundfile as sf
import pandas as pd
from pathlib import Path
import librosa
import sys
import numpy as np
import pdb
from rich.progress import track

wavdir = sys.argv[1]
txtdir = sys.argv[2]
thre_len = int(sys.argv[3])
origin_sr = int(sys.argv[4])
target_sr = int(sys.argv[5])

wavs = sorted(Path(wavdir).glob("**/*.wav"))
txts = sorted(Path(txtdir).glob("**/*.txt"))
target_dir = "./data/%s_%d_%d_len%d" % (
    Path(wavdir).stem,
    origin_sr,
    target_sr,
    thre_len,
)

Path.mkdir(Path(target_dir), exist_ok=True)
# pdb.set_trace()
tables = []
for x, y in track(
    zip(wavs, txts), description="Processing...", total=len(wavs)
):
    label = 1
    with open(y, "r") as f:
        txt = f.readline()
    if len(txt.split(" ")) <= thre_len:
        label = 1
        record = [x, Path(x).stem, txt, len(txt.split(" ")), label]
        tables.append(record)
    # Select length <= 10 words sentences for training
    if len(txt.split(" ")) <= thre_len:
        wav, sr = librosa.load(x, sr=origin_sr)
        wav_ = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
        sf.write(
            Path(target_dir) / Path((x).stem + ".wav"),
            data=wav_,
            samplerate=target_sr,
        )

D = pd.DataFrame(
    tables, columns=["wav_path", "id", "text", "len", "length_label"]
)
D.to_csv(target_dir + ".datalog", sep=",")
print("Check data log at %s" % (target_dir + ".datalog"))

D.get(["id", "text"]).to_csv(
    target_dir + ".txt", sep="\t", header=False, index=False, quoting=3
)

print("Generate id_text at %s" % (target_dir + ".txt"))
print("Finish")
