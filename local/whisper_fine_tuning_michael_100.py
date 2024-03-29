fine_tuning_dir = "fine_tuned/SSD/model/Michael_100_with_Large_AVA_script_conv_train_conv_dev"
"""
TODO:
    + [x] Load Configuration
    + [ ] Multi ASR Engine
    + [ ] Batch / Real Time support
"""
from pathlib import Path
from transformers import AutoTokenizer, AutoFeatureExtractor, AutoModelForCTC, AutoProcessor
from datasets import load_dataset, concatenate_datasets
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
audio_dir ="./data/Patient_sil_trim_16k_normed_5_snr_40"
healthy_dir="./data/Healthy"
Fary_PAL_30="./data/Fary_PAL_p326_20230110_30"
John_p326 = "./data/John_p326/output"
John_video = "./data/20230103_video"
p326_300_dir ="./data/John_p326_large"
negel_152 = "./data/4_negal_152_clean_all"

michael3_52 = "data/3_michael_20230619_100"

patient_T = "data/Patient_T/Patient_T"
patient_L = "data/Patient_L/Patient_L"
P1tony = "data/Participant1_Tony_Recording/CLEAN_SENTENCES/CONVERSATIONAL/PAL"
P1tony_arthur = "data/Participant1_Tony_Recording/CLEAN_SENTENCES/SCRIPTED/Arthur_the_Rat/PAL"
P1tony_rainbow = "data/Participant1_Tony_Recording/CLEAN_SENTENCES/SCRIPTED/Rainbow_Passage/Laronix"

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

# patient_L_test_dataset = load_dataset("audiofolder", data_dir=patient_L, split="train")
# patient_L_test_dataset = patient_L_test_dataset.map(dataclean)

# patient_T_test_dataset = load_dataset("audiofolder", data_dir=patient_T, split="train")
# patient_T_test_dataset = patient_T_test_dataset.map(dataclean)

P1tony_dataset = load_dataset("audiofolder", data_dir=P1tony, split="train")
P1tony_dataset = P1tony_dataset.map(dataclean)

P3Micheal_dataset_52 = load_dataset("audiofolder", data_dir=michael3_52, split="train")
P3Micheal_dataset_52 = P3Micheal_dataset_52.map(dataclean)

# negel_152_dataset = load_dataset("audiofolder", data_dir=negel_152, split="train")
# negel_152_dataset = negel_152_dataset.map(dataclean)


# pdb.set_trace()
# P1tony_scripted1 = load_dataset("audiofolder", data_dir=P1tony_rainbow, split="train")
# P1tony_scripted2 = load_dataset("audiofolder", data_dir=P1tony_arthur, split="train")
# P1tony_scripted1 = P1tony_scripted1.map(dataclean)
# P1tony_scripted2 = P1tony_scripted2.map(dataclean)
# P1tony_scripted = concatenate_datasets([P1tony_scripted1, P1tony_scripted2])

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
# from UV import plot_UV, get_speech_interval
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
p326_300_dataset = load_dataset("audiofolder", data_dir=p326_300_dir, split="train")
p326_300_dataset = p326_300_dataset.map(dataclean)

# healthy_test_dataset = load_dataset("audiofolder", data_dir=healthy_dir, split='train')
# healthy_test_dataset = healthy_test_dataset.map(dataclean)

# Fary_PAL_test_dataset = load_dataset("audiofolder", data_dir=Fary_PAL_30, split='train')
# Fary_PAL_test_dataset = Fary_PAL_test_dataset.map(dataclean)

# John_p326_test_dataset = load_dataset("audiofolder", data_dir=John_p326, split='train')
# John_p326_test_dataset = John_p326_test_dataset.map(dataclean)

# John_video_test_dataset = load_dataset("audiofolder", data_dir=John_video, split='train')
# John_video_test_dataset = John_video_test_dataset.map(dataclean)

# pdb.set_trace()

def train_dev_test_split(dataset: Dataset, dev_rate=0.1, test_rate=0.1, seed=1):
    """
    input: dataset
    dev_rate,
    test_rate
    seed
    -------
    Output:
    dataset_dict{"train", "dev", "test"}
    """
    train_dev_test = dataset.train_test_split(test_size=test_rate, seed=seed)
    test = train_dev_test["test"]
    train_dev = train_dev_test['train']
    
    # pdb.set_trace()
    if len(train_dev) <= int(len(dataset)*dev_rate):
        train = Dataset.from_dict({"audio": [], "transcription": []})
        dev = train_dev
    else:
        train_dev = train_dev.train_test_split(test_size=int(len(dataset)*dev_rate), seed=seed)
        train = train_dev['train']
        dev = train_dev['test']
    return train, dev, test

# pdb.set_trace()
# P1tony_train, P1tony_dev, P1tony_test = train_dev_test_split(P1tony_dataset, dev_rate=0.5, test_rate=0.5, seed=1)
# P1tony_train_ = concatenate_datasets([P1tony_train,P1tony_scripted]) 
# pdb.set_trace()

Michael_52_train, Michael_52_dev,  Michael_52_test = train_dev_test_split(P3Micheal_dataset_52, dev_rate=0.10, test_rate=0.1, seed=1)
 
# train_dev / test
ds = src_dataset.train_test_split(test_size=0.1, seed=1)

# dataset_libri = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

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
p326_encoded_train = p326_300_dataset.map(prepare_dataset, num_proc=4)

# combine large p326 in to training set
encoded_train = concatenate_datasets([encoded_train, p326_encoded_train])

# encoded_healthy = healthy_test_dataset.map(prepare_dataset, num_proc=4)
# encoded_Fary = Fary_PAL_test_dataset.map(prepare_dataset, num_proc=4)
# encoded_John_p326 = John_p326_test_dataset.map(prepare_dataset, num_proc=4)
# encoded_John_video = John_video_test_dataset.map(prepare_dataset, num_proc=4)

# encoded_P1tony_train = P1tony_train.map(prepare_dataset, num_proc=4)
# encoded_P1tony_dev = P1tony_dev.map(prepare_dataset, num_proc=4)
# encoded_P1tony_test = P1tony_test.map(prepare_dataset, num_proc=4)

# pdb.set_trace()
import numpy as np

WER = evaluate.load("wer")

## Whisper decoding

from transformers import WhisperProcessor, WhisperForConditionalGeneration, WhisperTokenizer, WhisperFeatureExtractor, Seq2SeqTrainingArguments, Seq2SeqTrainer, WhisperModel
processor = WhisperProcessor.from_pretrained("openai/whisper-medium")
# model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium").to("cuda:0")
model = WhisperForConditionalGeneration.from_pretrained("./fine_tuned/whipser_medium_en_PAL300_step25_step2_VCTK/checkpoint-400", use_auth_token=True).to("cuda:0")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-medium", language="English", task="transcribe")

from pathlib import Path
id = Path(fine_tuning_dir).stem
pdb.set_trace()
tokenizer.push_to_hub("KevinGeng/%s"%id)
# import pdb
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-medium")

def whisper_prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array 
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids 
    batch["labels"] = tokenizer(batch["transcription"]).input_ids
    return batch

torch.cuda.empty_cache()

def my_map_to_pred(batch):
    # pdb.set_trace()
    audio = batch["audio"]
    input_features = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt").input_features
    # batch["reference"] = whisper_processor.tokenizer._normalize(batch['text'])
    batch["reference"] = processor.tokenizer._normalize(batch['transcription'])

    with torch.no_grad():
        # predicted_ids = whisper_model.generate(input_features.to("cuda"))[0]
        predicted_ids = model.generate(input_features.to("cuda"))[0]
    transcription = model.decode(predicted_ids)
    batch["prediction"] = model.tokenizer._normalize(transcription)
    return batch

import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Union

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch
    
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * WER.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

# whisper_train = train.map(whisper_prepare_dataset, num_proc=4)
# pdb.set_trace()
whisper_train_large = encoded_train.map(whisper_prepare_dataset, num_proc=4)
whisper_dev = dev.map(whisper_prepare_dataset, num_proc=4)
whisper_test = test.map(whisper_prepare_dataset, num_proc=4)

encoded_Michael_52_train = Michael_52_train.map(whisper_prepare_dataset, num_proc=4)
encoded_Michael_52_dev = Michael_52_dev.map(whisper_prepare_dataset, num_proc=4)
encoded_Michael_52_test = Michael_52_test.map(whisper_prepare_dataset, num_proc=4)
# pdb.set_trace()
# # Add scirtped tony
# encoded_P1tony_train = P1tony_train_.map(whisper_prepare_dataset, num_proc=4)
# encoded_P1tony_dev = P1tony_dev.map(whisper_prepare_dataset, num_proc=4)
# encoded_P1tony_test = P1tony_test.map(whisper_prepare_dataset, num_proc=4)

# encode_negel_152_train = negel_152_train.map(whisper_prepare_dataset, num_proc=4)
# encode_negel_152_dev = negel_152_dev.map(whisper_prepare_dataset, num_proc=4)
# encode_negel_152_test = negel_152_test.map(whisper_prepare_dataset, num_proc=4)

# encoded_train_large = concatenate_datasets([whisper_train_large, encode_negel_152_train])
# encoded_dev_large = concatenate_datasets([whisper_dev, encode_negel_152_dev])

pdb.set_trace()
torch.cuda.empty_cache()

training_args = Seq2SeqTrainingArguments(
    output_dir=fine_tuning_dir,  # change to a repo name of your choice
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    warmup_steps=50,
    max_steps=1000,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    save_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=512,
    save_steps=10,
    eval_steps=10,
    logging_steps=10,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    save_total_limit=5,
    push_to_hub=False,
)
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback

# pdb.set_trace()
# # from transformers.trainer.callbacks import TensorBoardCallback
# class EvalLoggingCallback(TrainerCallback):
#     def on_evaluate(self, args, state, control, metrics, **kwargs):
#         print(f"Eval loss: {metrics['eval_loss']:.4f}, Accuracy: {metrics['eval_wer']:.4f}")
        
# pdb.set_trace()

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=encoded_Michael_52_train,
    eval_dataset=encoded_Michael_52_dev,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],

)
    # callbacks=[EvalLoggingCallback()]
trainer.train()
# trainer.evaluate(encoded_P1tony_test, metrix_key_prefix="test")
# trainer.callback_handler.on_test_end(trainer, datasets=encoded_P1tony_test)


# ## Not fine tuned
# z_result = encoded_test.map(my_map_to_pred)·
# # pdb.set_trace()
# # 0.4692737430167598
# z = WER.compute(references=z_result['reference'], predictions=z_result['prediction'])

# z_hel_result = encoded_healthy.map(my_map_to_pred)
# # 
# z_hel = WER.compute(references=z_hel_result['reference'], predictions=z_hel_result['prediction'])
# # 0.1591610117211598

# z_fary_result = encoded_Fary.map(my_map_to_pred)
# z_far = WER.compute(references=z_fary_result['reference'], predictions=z_fary_result['prediction'])
# # 0.1791044776119403


# z_john_p326_result = encoded_John_p326.map(my_map_to_pred)
# z_john_p326 = WER.compute(references=z_john_p326_result['reference'], predictions=z_john_p326_result['prediction'])
# # 0.4648241206030151

# # y_John_video= fine_tuned_trainer.predict(encoded_John_video)
# # metrics={'test_loss': 2.665189743041992, 'test_wer': 0.7222222222222222, 'test_runtime': 0.1633, 'test_samples_per_second': 48.979, 'test_steps_per_second': 6.122})
# pdb.set_trace()

# p326 training
# metrics={'test_loss': 0.4804028868675232, 'test_wer': 0.21787709497206703, 'test_runtime': 0.3594, 'test_samples_per_second': 44.517, 'test_steps_per_second': 5.565})
# hel metrics={'test_loss': 1.6363693475723267, 'test_wer': 0.17951881554595928, 'test_runtime': 3.8451, 'test_samples_per_second': 41.611, 'test_steps_per_second': 5.201})
# Fary: metrics={'test_loss': 1.4633615016937256, 'test_wer': 0.5572139303482587, 'test_runtime': 0.6627, 'test_samples_per_second': 45.27, 'test_steps_per_second': 6.036})
# p326 large: metrics={'test_loss': 0.6568527817726135, 'test_wer': 0.2889447236180904, 'test_runtime': 0.7169, 'test_samples_per_second': 51.613, 'test_steps_per_second': 6.975}) 