import sys
from pathlib import Path
root_path = Path(__file__).parents[1]
sys.path.append(str(root_path))

import argparse
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from random import sample
from sys import flags
import gradio as gr
import torchaudio
import torch
import torch.nn as nn
import src.lightning_module as lightning_module
import pdb
import jiwer
import numpy as np
# from tqdm import tqdm
from rich.progress import track

# Argparse
parser = argparse.ArgumentParser(
                    prog = 'ProgramName',
                    description = 'What the program does',
                    epilog = 'Text at the bottom of help')
parser.add_argument("--ref_txt", type=str, required=True)
parser.add_argument("--ref_wavs", type=str, required=True)
parser.add_argument("--target_dir", type=str, required=True)
parser.add_argument("--tag", type=str, default=None, required=False)
args = parser.parse_args()

refs = np.loadtxt(args.ref_txt, delimiter="\n", dtype="str")
refs_ids = [x.split(" ")[0] for x in refs]
refs_txt = [" ".join(x.split(" ")[1:]) for x in refs]
ref_wavs = [str(x) for x in sorted(
    Path(args.ref_wavs).glob("**/*.wav"))]
# pdb.set_trace()
try:
    len(refs) == len(ref_wavs)
except ValueError:
    print("Error: Text and Wavs don't match")
    exit()
    
# ASR part
from transformers import pipeline
p = pipeline("automatic-speech-recognition")

# WER part
transformation = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemoveWhiteSpace(replace_by_space=True),
    jiwer.RemoveMultipleSpaces(),
    jiwer.ReduceToListOfListOfWords(word_delimiter=" ")
])

# WPM part
processor = Wav2Vec2Processor.from_pretrained(
    "facebook/wav2vec2-xlsr-53-espeak-cv-ft")
phoneme_model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-xlsr-53-espeak-cv-ft")
# phoneme_model =  pipeline(model="facebook/wav2vec2-xlsr-53-espeak-cv-ft")


class ChangeSampleRate(nn.Module):
    def __init__(self, input_rate: int, output_rate: int):
        super().__init__()
        self.output_rate = output_rate
        self.input_rate = input_rate

    def forward(self, wav: torch.tensor) -> torch.tensor:
        # Only accepts 1-channel waveform input
        wav = wav.view(wav.size(0), -1)
        new_length = wav.size(-1) * self.output_rate // self.input_rate
        indices = (torch.arange(new_length) *
                   (self.input_rate / self.output_rate))
        round_down = wav[:, indices.long()]
        round_up = wav[:, (indices.long() + 1).clamp(max=wav.size(-1) - 1)]
        output = round_down * (1. - indices.fmod(1.)).unsqueeze(0) + \
            round_up * indices.fmod(1.).unsqueeze(0)
        return output


model = lightning_module.BaselineLightningModule.load_from_checkpoint(
    "epoch=3-step=7459.ckpt").eval()


def calc_mos(audio_path, ref):
    wav, sr = torchaudio.load(audio_path)
    osr = 16_000
    batch = wav.unsqueeze(0).repeat(10, 1, 1)
    csr = ChangeSampleRate(sr, osr)
    out_wavs = csr(wav)
    # ASR
    trans = p(audio_path)["text"]
    # WER
    wer = jiwer.wer(ref, trans, truth_transform=transformation,
                    hypothesis_transform=transformation)
    # MOS
    batch = {
        'wav': out_wavs,
        'domains': torch.tensor([0]),
        'judge_id': torch.tensor([288])
    }
    with torch.no_grad():
        output = model(batch)
    predic_mos = output.mean(dim=1).squeeze().detach().numpy()*2 + 3
    # Phonemes per minute (PPM)
    with torch.no_grad():
        logits = phoneme_model(out_wavs).logits
    phone_predicted_ids = torch.argmax(logits, dim=-1)
    phone_transcription = processor.batch_decode(phone_predicted_ids)
    lst_phonemes = phone_transcription[0].split(" ")
    wav_vad = torchaudio.functional.vad(wav, sample_rate=sr)
    ppm = len(lst_phonemes) / (wav_vad.shape[-1] / sr) * 60
    # if float(predic_mos) >= 3.0:
    #     torchaudio.save("good.wav", wav,sr)

    return predic_mos, trans, wer, phone_transcription, ppm


description = """
MOS prediction demo using UTMOS-strong w/o phoneme encoder model, which is trained on the main track dataset.
This demo only accepts .wav format. Best at 16 kHz sampling rate.

Paper is available [here](https://arxiv.org/abs/2204.02152)

Add ASR based on wav2vec-960, currently only English available.
Add WER interface.
"""

# # Auto load examples

# refs = np.loadtxt("Arthur_the_rat.txt", delimiter="\n", dtype="str")
# refs_ids = [x.split(" ")[0] for x in refs]
# refs_txt = [" ".join(x.split(" ")[1:]) for x in refs]
# ref_wavs = [str(x) for x in sorted(
#     Path("Patient_sil_trim_16k_normed_5_snr_40/Arthur_the_rat").glob("**/*.wav"))]

referance_id = gr.Textbox(value="ID",
                          placeholder="Utter ID",
                          label="Reference_ID")
referance_textbox = gr.Textbox(value="",
                               placeholder="Input reference here",
                               label="Reference")
# Set up interface
result = []
result.append("id, pred_mos, trans, wer, pred_phone, ppm")
for id, x, y in track(zip(refs_ids, ref_wavs, refs_txt), total=len(refs_ids), description="Loading references information"):
    predic_mos, trans, wer, phone_transcription, ppm = calc_mos(x, y)
    record = ",".join([id, str(predic_mos), str(trans), str(
        wer), str(phone_transcription), str(ppm)])
    result.append(record)

# Output
if args.tag == None:
    args.tag = Path(args.ref_wavs).stem
## Make output_dir
# pdb.set_trace()
Path.mkdir(Path(args.target_dir), exist_ok=True)
# pdb.set_trace()
with open("%s/%s.csv"%(args.target_dir, args.tag), "w") as f:
    print("\n".join(result), file=f)
