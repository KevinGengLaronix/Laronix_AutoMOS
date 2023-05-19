# Laronix Data Collection

This repository contains information about the Laronix data collection process, which involves collecting parallel data from AVA users. The dataset consists of two main sessions: scripted data and conversational data.

## Dataset

The dataset is organized as follows:

### 1. Scripted Data

The scripted data session includes 200 sentences collected from 5 articles. The references for both the audio and text versions of these sentences have already been uploaded or will be uploaded to the Laronix Recording system. (Ask [Kevin](kevin@laronix.com) for these files) The distribution of sentences from each article is as follows:

- Arthur the Rat: 56 sentences
- Cinder: 19 sentences
- Rainbow: 26 sentences
- Sentences: 59 sentences
- VCTK: 40 sentences

### 2. Conversational Data

The conversational data session focuses on natural conversations and involves the following components:

#### a. Q&A

In this component, a set of 50 sentences will be provided, consisting of questions and answers. During the recording, the partner will ask the questions (Q), and the patient will provide the answers (A). Both the questions and answers will be recorded.

#### b. Freestyle

The patients will have the freedom to talk about a given topic. They will be asked to respond with 5 to 10 sentences. The structure for this component can be referenced from the [IELTS speaking test](https://www.ieltsbuddy.com/IELTS-speaking-questions-with-answers.html).


## Document for Laronix Recording System

The Laronix recording system is designed for data collection from potential users of the AVA Device, which replaces their voice cord.

### Input:

- Audio signal
- Reference ID
- Reference text
- Reference Phoneme per minute

### Output:

- wav_pause_plot: Wave signal plot with pauses detected by VAD algorithm (SNR = 40dB)
- Predicted Mean Opinion Score: Score estimating data quality on the MOS scale using an ML prediction model (1-5)
- Hypotheses: Text predicted by Automatic Speech Recognition model (wav2vev2.0 + CTC)
- WER: Word Error Rate (lower is better)
- Predicted Phonemes
- PPM: Phonemes per minute
- Message: Feedback from the system

## User Instruction

Please follow the instructions provided at the top of the APP page.

```
- Laronix_AUTOMOS
    - data
        - Template
            - ref_wav/
                - 1.wav
                - 2.wav
                - ...
            - ref_txt.txt
            - ref.csv # audio prosody features reference <generate by script>
    - exp
        - Template
            - Audio_to_evaluate # RAW WAV DATA
            - log.csv # Recording log 
            - output # wav.file  <generate by script>
    - model
        - epoch=3-step=7459.ckpt # MOS estimate model
        - wav2vec_small.pt # WER model
    - local
        - get_ref_PPM.py # script for generating data/<ref_dir>/ref.csv
        - post_processing.py # script for generating exp/<ref_dir>/output/*.wav
```

---
title: Laronix Automos
emoji: 🏃
colorFrom: blue
colorTo: blue
sdk: gradio
sdk_version: 3.2
app_file: app.py
pinned: false
license: afl-3.0
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# Laronix_AutoMOS

## Usage:
### Step 1: Prepare data and text
`<todo>`
### Step 2: Preprocessing
```
## Generating *.csv, Voice/Unvoice Plot (optional) and config (optional)
python local/get_ref_PPM.py --ref_txt <ref_text> \
                            --ref_wavs <ref_wavs> \
                            --output_dir <output_dir> \
                            --to_config <True/False> \
                            --UV_flag <True/False> \
                            --UV_thre <UV_thre>}
```
### Step 3: Launch recording session:

```
## Start app.py
python app.py <config.yaml>
```
+ **Find logging below and lick URL to start**
```
Launch examples
Running on local URL:  http://127.0.0.1:7860/
...
    (Logs...)
...
Running on public URL: https://87abe771e93229da.gradio.app
```
