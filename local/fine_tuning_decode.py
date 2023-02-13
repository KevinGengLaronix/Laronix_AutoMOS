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
import librosa
# local import
import sys

sys.path.append("src")
import torch
torch.cuda.set_device("cuda:0")
# token_model = AutoModelForCTC.from_pretrained(
#     "facebook/wav2vec2-base-960h"
# )

# audio_dir= "/Users/kevingeng/Laronix/laronix_automos/data/Patient_sil_trim_16k_normed_5_snr_40/"
audio_dir ="/home/kevingeng/laronix/laronix_automos/data/Patient_sil_trim_16k_normed_5_snr_40"
healthy_dir="/home/kevingeng/laronix/laronix_automos/data/Healthy"
Fary_PAL_30="/home/kevingeng/laronix/laronix_automos/data/Fary_PAL_p326_20230110_30"
# audio_dir ="/home/kevingeng/laronix/laronix_automos/data/Healthy"
# tgt_audio_dir= "/Users/kevingeng/Laronix/Dataset/Pneumatic/automos"

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

import evaluate

wer = evaluate.load("wer")

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

# resample and clean text data
def dataclean(example):
    # pdb.set_trace()
    if example['audio']['sampling_rate'] != 16000:
        resampled_audio = librosa.resample(y=example['audio']['array'],
                                     orig_sr= example['audio']['sampling_rate'],
                                     target_sr=16000)
        # torchaudio.transforms.Resample(example['audio']['sampling_rate'], 16000)
        # resampled_audio = resampler(example['audio']['array'])

        return {"audio": {"path": example['audio']['path'], "array": resampled_audio, "sampling_rate": 16000},
                "transcription": example["transcription"].upper().translate(str.maketrans('', '', string.punctuation))}
    else:
        return {"transcription": example["transcription"].upper().translate(str.maketrans('', '', string.punctuation))}

# processor = AutoFeatureExtractor.from_pretrained(
#     "facebook/wav2vec2-base-960h"
# )
processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")

def prepare_dataset(batch):
    audio = batch["audio"]
    batch = processor(audio["array"], sampling_rate = audio["sampling_rate"], text=batch['transcription'])
    batch["input_length"] = len(batch["input_values"][0])
    return batch

src_dataset = load_dataset("audiofolder", data_dir=audio_dir, split="train")
src_dataset = src_dataset.map(dataclean)

healthy_test_dataset = load_dataset("audiofolder", data_dir=healthy_dir, split='train')
healthy_test_dataset = healthy_test_dataset.map(dataclean)

Fary_PAL_test_dataset = load_dataset("audiofolder", data_dir=Fary_PAL_30, split='train')
Fary_PAL_test_dataset = Fary_PAL_test_dataset.map(dataclean)
pdb.set_trace()

# train_dev / test
ds = src_dataset.train_test_split(test_size=0.1, seed=1)

train_dev = ds['train']
# train / dev
train_dev = train_dev.train_test_split(test_size=int(len(src_dataset)*0.1), seed=1)
# train/dev/test
train = train_dev['train']
test = ds['test']
dev = train_dev['test']

encoded_train = train.map(prepare_dataset, num_proc=4)
encoded_dev = dev.map(prepare_dataset, num_proc=4)
encoded_test = test.map(prepare_dataset, num_proc=4)

encoded_healthy = healthy_test_dataset.map(prepare_dataset, num_proc=4)
encoded_Fary = Fary_PAL_test_dataset.map(prepare_dataset, num_proc=4)
    # pdb.set_trace()
import numpy as np

WER = evaluate.load("wer")

import pdb
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

# data_collator
@dataclass
class DataCollatorCTCWithPadding:

    processor: AutoProcessor
    padding: Union[bool, str] = "longest"

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"][0]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        
        batch = self.processor.pad(input_features, padding=self.padding, return_tensors="pt")

        labels_batch = self.processor.pad(labels=label_features, padding=self.padding, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch

data_collator = DataCollatorCTCWithPadding(processor=processor, padding="longest")

def compute_metrics(pred):
    # pdb.set_trace()
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
    # pdb.set_trace()
    wer = WER.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

tokenizer = AutoTokenizer.from_pretrained("facebook/wav2vec2-base-960h")

from transformers import AutoModelForCTC, TrainingArguments, Trainer

model = AutoModelForCTC.from_pretrained(
    "facebook/wav2vec2-base-960h",
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
)

fine_tuned_model = AutoModelForCTC.from_pretrained(
    "PAL_John_128_train_dev_test_seed_1"
)

testing_args = TrainingArguments(
    output_dir="PAL_John_128_train_dev_test_seed_1",
    do_eval=True,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=1e-5,
    warmup_steps=0,
    max_steps=1000,
    gradient_checkpointing=True,
    fp16=True,
    group_by_length=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    save_steps=100,
    eval_steps=10,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
)

ori_trainer = Trainer(
    model=model,
    args=testing_args,
    tokenizer=processor.feature_extractor,
    compute_metrics=compute_metrics,
    data_collator=data_collator
)

fine_tuned_trainer = Trainer(
    model=fine_tuned_model,
    args=testing_args,
    tokenizer=processor.feature_extractor,
    compute_metrics=compute_metrics,
    data_collator=data_collator
)

x = ori_trainer.predict(encoded_test)
# {'eval_loss': 2.157982349395752, 'eval_wer': 0.8491620111731844, 'eval_runtime': 0.3919, 'eval_samples_per_second': 40.824, 'eval_steps_per_second': 5.103}
x_hel = ori_trainer.predict(encoded_healthy)
# {'eval_loss': 1.6561158895492554, 'eval_wer': 0.19925971622455274, 'eval_runtime': 3.7416, 'eval_samples_per_second': 42.762, 'eval_steps_per_second': 5.345}
x_fary = ori_trainer.predict(encoded_Fary)
# {'eval_loss': 1.8639644384384155, 'eval_wer': 0.43781094527363185, 'eval_runtime': 0.6672, 'eval_samples_per_second': 44.964, 'eval_steps_per_second': 5.995}
y = fine_tuned_trainer.predict(encoded_test)
# {'eval_loss': 0.6034298539161682, 'eval_wer': 0.2011173184357542, 'eval_runtime': 0.3675, 'eval_samples_per_second': 43.541, 'eval_steps_per_second': 5.443}
y_hel = fine_tuned_trainer.predict(encoded_healthy)
# {'eval_loss': 1.7949535846710205, 'eval_wer': 0.1566933991363356, 'eval_runtime': 3.665, 'eval_samples_per_second': 43.656, 'eval_steps_per_second': 5.457}
y_fary = fine_tuned_trainer.predict(encoded_Fary)
# {'eval_loss': 1.7337630987167358, 'eval_wer': 0.5472636815920398, 'eval_runtime': 0.6353, 'eval_samples_per_second': 47.219, 'eval_steps_per_second': 6.296}
pdb.set_trace()
