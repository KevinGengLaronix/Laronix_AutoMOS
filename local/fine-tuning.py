"""
TODO:
    + [x] Load Configuration
    + [ ] Multi ASR Engine
    + [ ] Batch / Real Time support
"""
from pathlib import Path
from transformers import AutoTokenizer, AutoFeatureExtractor, AutoModelForCTC, AutoProcessor
from datasets import load_dataset
from datasets import Dataset, Audio
import pdb
import string
# local import
import sys

sys.path.append("src")

# token_model = AutoModelForCTC.from_pretrained(
#     "facebook/wav2vec2-base-960h"
# )

# ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")

audio_path = "/Users/kevingeng/Laronix/Laronix_PAL_ASR_Offline_Plot/data/samples/3_Healthy1.wav"

audio_dir= "/Users/kevingeng/Laronix/laronix_automos/data/Patient_sil_trim_16k_normed_5_snr_40/"
# tgt_audio_dir= "/Users/kevingeng/Laronix/Dataset/Pneumatic/automos"

# src_audio_list = sorted(Path(src_audio_dir).glob("**/*.wav"))
# src_audio_list = [str(x) for x in src_audio_list]
# src_audio_dict = {"audio": src_audio_list}
# src_dataset = Dataset.from_dict(src_audio_dict).cast_column("audio", Audio())

# tgt_audio_list = sorted(Path(tgt_audio_dir).glob("**/*.wav"))
# tgt_audio_list = [str(x) for x in tgt_audio_list]
# tgt_audio_dict = {"audio": tgt_audio_list}
# tgt_dataset = Dataset.from_dict(tgt_audio_dict).cast_column("audio", Audio())

# Get Transcription, WER and PPM 
"""
TODO:
    [DONE]: Automatic generating Config
"""

import yaml
import argparse
import sys
from pathlib import Path

sys.path.append("./src")
import lightning_module
from UV import plot_UV, get_speech_interval
from transformers import pipeline
from rich.progress import track
from rich import print as rprint
import numpy as np
import jiwer
import pdb
import torch.nn as nn
import torch
import torchaudio
import gradio as gr
from sys import flags
from random import sample
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# root_path = Path(__file__).parents[1]

class ChangeSampleRate(nn.Module):
    def __init__(self, input_rate: int, output_rate: int):
        super().__init__()
        self.output_rate = output_rate
        self.input_rate = input_rate

    def forward(self, wav: torch.tensor) -> torch.tensor:
        # Only accepts 1-channel waveform input
        wav = wav.view(wav.size(0), -1) 
        new_length = wav.size(-1) * self.output_rate // self.input_rate
        indices = torch.arange(new_length) * (
            self.input_rate / self.output_rate
        )
        round_down = wav[:, indices.long()]
        round_up = wav[:, (indices.long() + 1).clamp(max=wav.size(-1) - 1)]
        output = round_down * (1.0 - indices.fmod(1.0)).unsqueeze(
            0
        ) + round_up * indices.fmod(1.0).unsqueeze(0)
        return output


model = lightning_module.BaselineLightningModule.load_from_checkpoint(
    "./src/epoch=3-step=7459.ckpt"
).eval()


def calc_wer(audio_path, ref):
    wav, sr = torchaudio.load(audio_path)
    osr = 16_000
    batch = wav.unsqueeze(0).repeat(10, 1, 1)
    csr = ChangeSampleRate(sr, osr)
    out_wavs = csr(wav)
    # ASR
    trans = p(audio_path)["text"]
    # WER
    wer = jiwer.wer(
        ref,
        trans,
        truth_transform=transformation,
        hypothesis_transform=transformation,
    )
    return trans, wer

# if __name__ == "__main__":
#     # Argparse
#     parser = argparse.ArgumentParser(
#         prog="get_ref_PPM",
#         description="Generate Phoneme per Minute (and Voice/Unvoice plot)",
#         epilog="",
#     )
#     parser.add_argument(
#         "--tag",
#         type=str,
#         default=None,
#         required=False,
#         help="ID tag for output *.csv",
#     )

#     parser.add_argument("--ref_txt", type=str, required=True, help="Reference TXT")
#     parser.add_argument(
#         "--ref_wavs", type=str, required=True, help="Reference WAVs"
#     )

#     parser.add_argument(
#         "--output_dir",
#         type=str,
#         required=True,
#         help="Output Directory for *.csv",
#     )
#     parser.add_argument(
#         "--to_config",
#         choices=["True", "False"],
#         default="False",
#         help="Generating Config from .txt and wavs/*wav",
#     )


#     args = parser.parse_args()

#     refs = np.loadtxt(args.ref_txt, delimiter="\n", dtype="str")
#     refs_ids = [x.split()[0] for x in refs]
#     refs_txt = [" ".join(x.split()[1:]) for x in refs]
#     ref_wavs = [str(x) for x in sorted(Path(args.ref_wavs).glob("**/*.wav"))]
#     # pdb.set_trace()
#     try:
#         len(refs) == len(ref_wavs)
#     except ValueError:
#         print("Error: Text and Wavs don't match")
#         exit()

#     # ASR part
#     p = pipeline("automatic-speech-recognition")

#     # WER part
#     transformation = jiwer.Compose(
#         [
#             jiwer.ToLowerCase(),
#             jiwer.RemoveWhiteSpace(replace_by_space=True),
#             jiwer.RemoveMultipleSpaces(),
#             jiwer.ReduceToListOfListOfWords(word_delimiter=" "),
#         ]
#     )

#     # WPM part
#     processor = Wav2Vec2Processor.from_pretrained(
#         "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
#     )
#     phoneme_model = Wav2Vec2ForCTC.from_pretrained(
#         "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
#     )
#     # phoneme_model =  pipeline(model="facebook/wav2vec2-xlsr-53-espeak-cv-ft")

#     description = """
#     MOS prediction demo using UTMOS-strong w/o phoneme encoder model, \
#         which is trained on the main track dataset.
#         This demo only accepts .wav format. Best at 16 kHz sampling rate.

#     Paper is available [here](https://arxiv.org/abs/2204.02152)

#     Add ASR based on wav2vec-960, currently only English available.
#     Add WER interface.
#     """

#     referance_id = gr.Textbox(
#         value="ID", placeholder="Utter ID", label="Reference_ID"
#     )
#     referance_textbox = gr.Textbox(
#         value="", placeholder="Input reference here", label="Reference"
#     )
#     # Set up interface
#     result = []
#     result.append("id, trans, wer")

    
#     for id, x, y in track(
#         zip(refs_ids, ref_wavs, refs_txt),
#         total=len(refs_ids),
#         description="Loading references information",
#     ):
#         trans, wer = calc_wer(x, y)
#         record = ",".join(
#             [
#                 id,
#                 str(trans),
#                 str(wer)
#             ]
#         )
#         result.append(record)

#     # Output
#     if args.tag == None:
#         args.tag = Path(args.ref_wavs).stem
#     # Make output_dir
#     # pdb.set_trace()
#     Path.mkdir(Path(args.output_dir), exist_ok=True)
#     # pdb.set_trace()
#     with open("%s/%s.csv" % (args.output_dir, args.tag), "w") as f:
#         print("\n".join(result), file=f)

#     # Generating config
#     if args.to_config == "True":
#         config_dict = {
#             "exp_id": args.tag,
#             "ref_txt": args.ref_txt,
#             "ref_feature": "%s/%s.csv" % (args.output_dir, args.tag),
#             "ref_wavs": args.ref_wavs,
#             "thre": {
#                 "minppm": 100,
#                 "maxppm": 100,
#                 "WER": 0.1,
#                 "AUTOMOS": 4.0,
#             },
#             "auth": {"username": None, "password": None},
#         }
#         with open("./config/%s.yaml" % args.tag, "w") as config_f:
#             rprint("Dumping as config ./config/%s.yaml" % args.tag)
#             rprint(config_dict)
#             yaml.dump(config_dict, stream=config_f)
#             rprint("Change parameter ./config/%s.yaml if necessary" % args.tag)
#     print("Reference Dumping Finished")
def dataclean(example):
    return {"transcription": example["transcription"].upper().translate(str.maketrans('', '', string.punctuation))}

# processor = AutoFeatureExtractor.from_pretrained(
#     "facebook/wav2vec2-base-960h"
# )
processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base")

def prepare_dataset(batch):
    audio = batch["audio"]
    batch = processor(audio["array"], sampling_rate = audio["sampling_rate"], text=batch['transcription'])
    batch["input_length"] = len(batch["input_values"][0])
    return batch

src_dataset = load_dataset("audiofolder", data_dir=audio_dir, split="train")
src_dataset = src_dataset.map(dataclean)
# train_dev / test
ds = src_dataset.train_test_split(test_size=0.1)

train_dev = ds['train']
# train / dev
train_dev = train_dev.train_test_split(test_size=int(len(src_dataset)*0.1))
# train/dev/test
train = train_dev['train']
test = ds['test']
dev = train_dev['test']

    # pdb.set_trace()
import numpy as np


def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


pdb.set_trace()
# TOKENLIZER("data/samples/5_Laronix1.wav")
# pdb.set_trace()
# tokenizer 
tokenizer = AutoTokenizer.from_pretrained("facebook/wav2vec2-base-960h")

encoded_train = train.map(prepare_dataset, num_proc=4)

from transformers import AutoModelForCTC, TrainingArguments, Trainer

model = AutoModelForCTC.from_pretrained(
    "facebook/wav2vec2-base",
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
)
pdb.set_trace()

training_args = TrainingArguments(
    output_dir="my_awesome_asr_mind_model",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=2000,
    gradient_checkpointing=True,
    fp16=True,
    group_by_length=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=True,
)

pdb.set_trace()
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_train["train"],
    eval_dataset=encoded_train["test"],
    tokenizer=processor.feature_extractor,
    compute_metrics=compute_metrics,
)
pdb.set_trace()
    # data_collator=data_collator,

trainer.train()
# x = tokenizer(test['transcription'][0])

