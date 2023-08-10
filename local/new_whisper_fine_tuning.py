fine_tuning_dir = "/fine_tuned/whipser_medium_en_PAL300_step25_step2_VCTK/checkpoint-400"

"""
TODO:
    + [ ] Data load
    + [ ] Train / Test / Dev spilt 
    + [ ] Train / Test Phase
    + [ ] Logging with Train / Dev / Test Loss
    + [ ] Evalutation metrics
"""
import pdb
import string
from pathlib import Path

import evaluate
import librosa
import torch
import torch.nn as nn
from datasets import Dataset, concatenate_datasets, load_dataset
from transformers import AutoProcessor

wer = evaluate.load("wer")
torch.cuda.set_device("cuda:0")

audio_dir = "./data/Patient_sil_trim_16k_normed_5_snr_40"
healthy_dir = "./data/Healthy"
Fary_PAL_30 = "./data/Fary_PAL_p326_20230110_30"
John_p326 = "./data/John_p326/output"
John_video = "./data/20230103_video"

## train
p326_300_dir = "./data/John_p326_large"
P1tony_arthur = "data/Participant1_Tony_Recording/CLEAN_SENTENCES/SCRIPTED/Arthur_the_Rat/PAL"
P1tony_rainbow = "data/Participant1_Tony_Recording/CLEAN_SENTENCES/SCRIPTED/Rainbow_Passage/Laronix"

P1tony = "data/Participant1_Tony_Recording/CLEAN_SENTENCES/CONVERSATIONAL/PAL"

P4Negel = 'data/4_negal_152_clean_all'

def dataclean(example):
    if example["audio"]["sampling_rate"] != 16000:
        resampled_audio = librosa.resample(
            y=example["audio"]["array"],
            orig_sr=example["audio"]["sampling_rate"],
            target_sr=16000,
        )

        return {
            "audio": {
                "path": example["audio"]["path"],
                "array": resampled_audio,
                "sampling_rate": 16000,
            },
            "transcription": example["transcription"]
            .upper()
            .translate(str.maketrans("", "", string.punctuation)),
        }
    else:
        return {
            "transcription": example["transcription"]
            .upper()
            .translate(str.maketrans("", "", string.punctuation))
        }



P1tony_dataset = load_dataset("audiofolder", data_dir=P1tony, split="train")
P1tony_dataset = P1tony_dataset.map(dataclean)

P1tony_scripted1 = load_dataset(
    "audiofolder", data_dir=P1tony_rainbow, split="train"
)
P1tony_scripted2 = load_dataset(
    "audiofolder", data_dir=P1tony_arthur, split="train"
)
P1tony_scripted1 = P1tony_scripted1.map(dataclean)
P1tony_scripted2 = P1tony_scripted2.map(dataclean)
P1tony_scripted = concatenate_datasets([P1tony_scripted1, P1tony_scripted2])

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
    if example["audio"]["sampling_rate"] != 16000:
        resampled_audio = librosa.resample(
            y=example["audio"]["array"],
            orig_sr=example["audio"]["sampling_rate"],
            target_sr=16000,
        )
        
        return {
            "audio": {
                "path": example["audio"]["path"],
                "array": resampled_audio,
                "sampling_rate": 16000,
            },
            "transcription": example["transcription"]
            .upper()
            .translate(str.maketrans("", "", string.punctuation)),
        }
    else:
        return {
            "transcription": example["transcription"]
            .upper()
            .translate(str.maketrans("", "", string.punctuation))
        }


# processor = AutoFeatureExtractor.from_pretrained(
#     "facebook/wav2vec2-base-960h"
# )
processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")

def prepare_dataset(batch):
    audio = batch["audio"]
    batch = processor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
        text=batch["transcription"],
    )
    batch["input_length"] = len(batch["input_values"][0])
    return batch

src_dataset = load_dataset("audiofolder", data_dir=audio_dir, split="train")
src_dataset = src_dataset.map(dataclean)
p326_300_dataset = load_dataset(
    "audiofolder", data_dir=p326_300_dir, split="train"
)
p326_300_dataset = p326_300_dataset.map(dataclean)

P4Negel_dataset = load_dataset("audiofolder", data_dir=P4Negel, split="train")
P4Negel_dataset = P4Negel_dataset.map(dataclean)

healthy_test_dataset = load_dataset(
    "audiofolder", data_dir=healthy_dir, split="train"
)
healthy_test_dataset = healthy_test_dataset.map(dataclean)

Fary_PAL_test_dataset = load_dataset(
    "audiofolder", data_dir=Fary_PAL_30, split="train"
)
Fary_PAL_test_dataset = Fary_PAL_test_dataset.map(dataclean)

John_p326_test_dataset = load_dataset(
    "audiofolder", data_dir=John_p326, split="train"
)
John_p326_test_dataset = John_p326_test_dataset.map(dataclean)

John_video_test_dataset = load_dataset(
    "audiofolder", data_dir=John_video, split="train"
)
John_video_test_dataset = John_video_test_dataset.map(dataclean)


def train_dev_test_split(
    dataset: Dataset, dev_rate=0.1, test_rate=0.1, seed=1
):
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
    train_dev = train_dev_test["train"]

    # pdb.set_trace()
    if len(train_dev) <= int(len(dataset) * dev_rate):
        train = Dataset.from_dict({"audio": [], "transcription": []})
        dev = train_dev
    else:
        train_dev = train_dev.train_test_split(
            test_size=int(len(dataset) * dev_rate), seed=seed
        )
        train = train_dev["train"]
        dev = train_dev["test"]
    return train, dev, test

P1tony_train, P1tony_dev, P1tony_test = train_dev_test_split(
    P1tony_dataset, dev_rate=0.5, test_rate=0.5, seed=1
)
P1tony_train_ = concatenate_datasets([P1tony_train, P1tony_scripted])

# train_dev / test
ds = src_dataset.train_test_split(test_size=0.1, seed=1)

# dataset_libri = load_dataset(
#     "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation"
# )

train_dev = ds["train"]
# train / dev
train_dev = train_dev.train_test_split(
    test_size=int(len(src_dataset) * 0.1), seed=1
)

# Tony
Tony_train = P1tony_train_
Tony_dev = P1tony_dev
Tony_test = P1tony_test

# John
John_train, John_dev, John_test = train_dev_test_split(p326_300_dataset, dev_rate=0.1, test_rate=0.1)
# Negel
Negel_train, Negel_dev, Negel_test = train_dev_test_split(P4Negel_dataset, dev_rate=0.1, test_rate=0.1)

# train/dev/test
train = train_dev["train"]
test = ds["test"]
dev = train_dev["test"]

# combined
combine_train = concatenate_datasets([train, Tony_train, John_train, Negel_train])
conbine_dev = concatenate_datasets([dev, Tony_dev, John_dev, Negel_dev])
conbine_test = concatenate_datasets([test, Tony_test, John_test, Negel_test])

# encoded_train = combine_train.map(prepare_dataset, num_proc=4)
# encoded_dev = conbine_dev.map(prepare_dataset, num_proc=4)
# encoded_test = conbine_test.map(prepare_dataset, num_proc=4)

# # extra_test
# encoded_Fary = Fary_PAL_test_dataset.map(prepare_dataset, num_proc=4)
# encoded_healthy = healthy_test_dataset.map(prepare_dataset, num_proc=4)

# encoded_ori_test = test.map(prepare_dataset, num_proc=4)
# encoded_Tony_test = Tony_test.map(prepare_dataset, num_proc=4)
# encoded_John_test = John_test.map(prepare_dataset, num_proc=4)
# encoded_Negel_test = Negel_test.map(prepare_dataset, num_proc=4)

# encoded_train = train.map(prepare_dataset, num_proc=4)
# encoded_dev = dev.map(prepare_dataset, num_proc=4)
# p326_encoded_train = p326_300_dataset.map(prepare_dataset, num_proc=4)

# combine large p326 in to training set
# encoded_train = concatenate_datasets([encoded_train, p326_encoded_train])

# encoded_John_p326 = John_p326_test_dataset.map(prepare_dataset, num_proc=4)
# encoded_John_video = John_video_test_dataset.map(prepare_dataset, num_proc=4)

# pdb.set_trace()
import numpy as np

WER = evaluate.load("wer")

## Whisper decoding

from transformers import (Seq2SeqTrainer, Seq2SeqTrainingArguments,
                          WhisperFeatureExtractor,
                          WhisperForConditionalGeneration, WhisperModel,
                          WhisperProcessor, WhisperTokenizer)

processor = WhisperProcessor.from_pretrained("openai/whisper-medium")
# model = WhisperForConditionalGeneration.from_pretrained(
#     "./fine_tuned/whipser_medium_en_PAL300_step25_step2_VCTK/checkpoint-400",
#     use_auth_token=True,
# ).to("cuda:0")
model = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-medium",
).to("cuda:0")
tokenizer = WhisperTokenizer.from_pretrained(
    "openai/whisper-medium", language="English", task="transcribe"
)

from pathlib import Path

id = Path(fine_tuning_dir).stem
pdb.set_trace()
tokenizer.push_to_hub("KevinGeng/%s" % id)
# import pdb
feature_extractor = WhisperFeatureExtractor.from_pretrained(
    "openai/whisper-medium"
)

def whisper_prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]

    # encode target text to label ids
    batch["labels"] = tokenizer(batch["transcription"]).input_ids
    return batch

torch.cuda.empty_cache()


def my_map_to_pred(batch):
    # pdb.set_trace()
    audio = batch["audio"]
    input_features = processor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
        return_tensors="pt",
    ).input_features
    # batch["reference"] = whisper_processor.tokenizer._normalize(batch['text'])
    batch["reference"] = processor.tokenizer._normalize(batch["transcription"])

    with torch.no_grad():
        # predicted_ids = whisper_model.generate(input_features.to("cuda"))[0]
        predicted_ids = model.generate(input_features.to("cuda"))[0]
    transcription = model.decode(predicted_ids)
    batch["prediction"] = model.tokenizer._normalize(transcription)
    return batch


from dataclasses import dataclass
from typing import Any, Dict, List, Union

import torch


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [
            {"input_features": feature["input_features"]}
            for feature in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        # get the tokenized label sequences
        label_features = [
            {"input_ids": feature["labels"]} for feature in features
        ]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (
            (labels[:, 0] == self.processor.tokenizer.bos_token_id)
            .all()
            .cpu()
            .item()
        ):
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

encoded_train = combine_train.map(whisper_prepare_dataset, num_proc=4)
encoded_dev = conbine_dev.map(whisper_prepare_dataset, num_proc=4)
encoded_test = conbine_test.map(whisper_prepare_dataset, num_proc=4)

# extra_test

encoded_ori_test = test.map(whisper_prepare_dataset, num_proc=4)
encoded_Tony_test = Tony_test.map(whisper_prepare_dataset, num_proc=4)
encoded_John_test = John_test.map(whisper_prepare_dataset, num_proc=4)
encoded_Negel_test = Negel_test.map(whisper_prepare_dataset, num_proc=4)

encoded_Fary = Fary_PAL_test_dataset.map(whisper_prepare_dataset, num_proc=4)
encoded_healthy = healthy_test_dataset.map(whisper_prepare_dataset, num_proc=4)

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
    save_steps=20,
    eval_steps=20,
    logging_steps=10,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    save_total_limit=5,
    push_to_hub=False,
)
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=Negel_train,
    eval_dataset=Negel_dev,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
)
# callbacks=[EvalLoggingCallback()]
pdb.set_trace()

before_result_dict = {
    "Ori_Test": trainer.evaluate(encoded_ori_test),
    "Tony_Test": trainer.evaluate(encoded_Tony_test),
    "John_Test": trainer.evaluate(encoded_John_test),
    "Negel_Test": trainer.evaluate(encoded_Negel_test),
    "Zeroshot_Fary_Test": trainer.evaluate(encoded_Fary),
    "Healthy_Test": trainer.evaluate(encoded_healthy),
}

print(before_result_dict)
trainer.train()

pdb.set_trace()
result_dict = {
    "Ori_Test": trainer.evaluate(encoded_ori_test),
    "Tony_Test": trainer.evaluate(encoded_Tony_test),
    "John_Test": trainer.evaluate(encoded_John_test),
    "Negel_Test": trainer.evaluate(encoded_Negel_test),
    "Zeroshot_Fary_Test": trainer.evaluate(encoded_Fary),
    "Healthy_Test": trainer.evaluate(encoded_healthy),
}

pdb.set_trace()
# Evaluation
model.push_to_hub("KevinGeng/%s" % id)