'''
TODO:
    + [ ] Load Configuration
    + [ ] Checking
    + [ ] Better saving directory
'''

from transformers import pipeline
import yaml
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from logging import PlaceHolder
import gradio as gr
import torchaudio
import torch
import torch.nn as nn
import src.lightning_module
import pdb
import jiwer
from pathlib import Path
import numpy as np
import sys
import librosa
import librosa.display
import matplotlib.pyplot as plt

# Load automos
config_yaml = sys.argv[1]
with open(config_yaml, "r") as f:
    pdb.set_trace()
    try:
        config = yaml.safe_load(f)
    except FileExistsError:
        print("Config file Loading Error")
        exit()

# Auto load examples
refs = np.loadtxt(config["ref_txt"], delimiter="\n", dtype="str")
refs_ids = [x.split(" ")[0] for x in refs]
refs_txt = [" ".join(x.split(" ")[1:]) for x in refs]
ref_feature = np.loadtxt(config["ref_feature"], delimiter=",", dtype="str")
ref_wavs = [str(x) for x in sorted(Path(config["ref_wavs"]).glob("**/*.wav"))]

refs_ppm = np.array(ref_feature[:, -1][1:], dtype="str")
reference_id = gr.Textbox(value="ID",
                          placeholder="Utter ID",
                          label="Reference_ID")

reference_textbox = gr.Textbox(value="Input reference here",
                               placeholder="Input reference here",
                               label="Reference")
reference_PPM = gr.Textbox(
    placeholder="Pneumatic Voice's PPM", label="Ref PPM")
# Flagging setup

# Set up interface
print("Preparing Examples")
examples = [[w, i, x, y]
            for w, i, x, y in zip(ref_wavs, refs_ids, refs_txt, refs_ppm)]

# ASR part
p = pipeline("automatic-speech-recognition")

# WER part
transformation = jiwer.Compose([
    jiwer.RemovePunctuation(),
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


# MOS model
model = lightning_module.BaselineLightningModule.load_from_checkpoint(
    "src/epoch=3-step=7459.ckpt").eval()

# Get Speech Interval


def get_speech_interval(signal, db):
    audio_interv = librosa.effects.split(signal, top_db=db)
    pause_end = [x[0] for x in audio_interv[1:]]
    pause_start = [x[1] for x in audio_interv[0: -1]]
    pause_interv = [[x, y] for x, y in zip(pause_start, pause_end)]
    return audio_interv, pause_interv

# plot UV


def plot_UV(signal, audio_interv, sr):
    fig, ax = plt.subplots(nrows=2, sharex=True)
    librosa.display.waveshow(signal, sr=sr, ax=ax[0])
    uv_flag = np.zeros(len(signal))
    for i in audio_interv:
        uv_flag[i[0]: i[1]] = 1

    ax[1].plot(np.arange(len(signal))/sr, uv_flag, "r")
    ax[1].set_ylim([-0.1,  1.1])
    return fig

# Evaluation model


def calc_mos(audio_path, id, ref, pre_ppm, fig):
    wav, sr = torchaudio.load(audio_path)
    osr = 16000
    batch = wav.unsqueeze(0).repeat(10, 1, 1)
    csr = ChangeSampleRate(sr, osr)
    out_wavs = csr(wav)
    # ASR
    trans = jiwer.ToLowerCase()(p(audio_path)["text"])

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

    a_h, p_h = get_speech_interval(wav_vad.numpy(), db=40)
    print(a_h)
    print(len(a_h))
    fig_h = plot_UV(wav_vad.numpy().squeeze(), a_h, sr=sr)
    ppm = len(lst_phonemes) / (wav_vad.shape[-1] / sr) * 60
    # if float(predic_mos) >= 3.0:
    #     torchaudio.save("good.wav", wav,sr)

    error_msg = "!!! ERROR MESSAGE !!!\n"
    if ppm >= float(pre_ppm) + float(config["thre"]["maxppm"]):
        error_msg += "ERROR: Please speak slower.\n"
    elif ppm <= float(pre_ppm) + float(config["thre"]["minppm"]):
        error_msg += "ERROR: Please speak faster.\n"
    elif predic_mos <= float(config["thre"]["AUTOMOS"]):
        error_msg += "ERROR: Naturalness is too low, Please try again.\n"
    elif wer >= float(config["thre"]["WER"]):
        error_msg += "ERROR: Intelliablity is too low, Please try again\n"
    else:
        error_msg = "Good JOB! Please click the Flag Button to save this record.\n You can start recording the next one."

    return predic_mos, trans, wer, phone_transcription, ppm, fig_h, error_msg


# Description
description = """
This is the experiment page for Laronix Data Recording.\n
\n
1. Select one example from below, a Pneumatic Voice file, a transcription, and its speaking rate will be loaded as inputs.\n
    You can check the pneumatic samples first and prepare for reading the transcription at a similar tempo.\n
2. Delete the pneumatic voice (click the X button on the right), a recording button will appear.\n
3. Click the recording button to start recording, click again to stop. Make sure you are not mispronouncing or include any detectable noises.\n
4. Click "Submit" button and wait for the result.\n
5. Please check the message box to see the feedback, if ERROR appears, delete your previous recording and try again :).\n
6. If "Good JOB!" message appears, click "Flag as Perfect" and start another recording.\n
7. If you try several times (N >= 10) and still can not clear the mission, you can flag your best recording by clicking "Doubtful Speaking Rate" or "Doubtful Naturalness". \n
    Yet this seldom happens, so please try to meet the system's requirement first!\n
8. If you have any other question, Please contact kevin@laronix.com \n

Requirements: \n

Predicted MOS >=%f \n
WER <= %f \n
Pef PPM - %f < PPM < Ref PPM + %f \n
""" % (float(config["thre"]["AUTOMOS"]), float(config["thre"]["WER"]), -float(config["thre"]["minppm"]), float(config["thre"]["maxppm"]))

# Auto load examples
refs = np.loadtxt("p326_split.txt", delimiter="\n", dtype="str")
refs_ids = [x.split(" ")[0] for x in refs]
refs_txt = [" ".join(x.split(" ")[1:]) for x in refs]
ref_feature = np.loadtxt("p326_split_ref.csv", delimiter=",", dtype="str")
ref_wavs = [str(x) for x in sorted(Path("p326_split").glob("**/*.wav"))]

refs_ppm = np.array(ref_feature[:, -1][1:], dtype="str")
reference_id = gr.Textbox(value="ID",
                          placeholder="Utter ID",
                          label="Reference_ID")
reference_textbox = gr.Textbox(value="Input reference here",
                               placeholder="Input reference here",
                               label="Reference")
reference_PPM = gr.Textbox(
    placeholder="Pneumatic Voice's PPM", label="Ref PPM")

# Flagging setup

# Set up interface
print("Preparing Examples")
examples = [[w, i, x, y]
            for w, i, x, y in zip(ref_wavs, refs_ids, refs_txt, refs_ppm)]

# Interface
# Participant Information


def record_part_info(name, gender, first_lng):
    message = "Participant information is successfully collected."
    id_str = "%s_%s_%s" % (name, gender[0], first_lng[0])

    if name == None:
        message = "ERROR: Name Information imcompleted!"
        id_str = "ERROR"

    if gender == None:
        message = "ERROR: Please select gender"
        id_str = "ERROR"

    if len(gender) > 1:
        message = "ERROR: Please select one gender only"
        id_str = "ERROR"

    if first_lng == None:
        message = "ERROR: Please select your english proficiency"
        id_str = "ERROR"

    if len(first_lng) > 1:
        message = "ERROR: Please select one english proficienty only"
        id_str = "ERROR"

    return message, id_str


# infomation page
name = gr.Textbox(placeholder="Name", label="Name")
gender = gr.CheckboxGroup(["Male", "Female"], label="gender")
first_lng = gr.CheckboxGroup(
    ["English", "Others"], label="English Proficiency")

msg = gr.Textbox(
    placeholder="Evalutation for valid participant", label="message")
id_str = gr.Textbox(placeholder="participant id", label="participant_id")

info = gr.Interface(fn=record_part_info,
                    inputs=[name, gender, first_lng],
                    outputs=[msg, id_str],
                    title="Participant information Page",
                    allow_flagging="manual",
                    css="body {background-color: blue}"
                    )
# Experiment

iface = gr.Interface(
    fn=calc_mos,
    inputs=[gr.Audio(source="microphone", type='filepath', label="Audio_to_evaluate"),
            reference_id,
            reference_textbox,
            reference_PPM,
            gr.Image(type="filepath", label="Ref Wav and Pauses")],
    outputs=[gr.Textbox(placeholder="Predicted MOS", label="Predicted MOS"),
             gr.Textbox(placeholder="Hypothesis", label="Hypothesis"),
             gr.Textbox(placeholder="Word Error Rate", label="WER"),
             gr.Textbox(placeholder="Predicted Phonemes",
                        label="Predicted Phonemes"),
             gr.Textbox(placeholder="Phonemes per minutes", label="PPM"),
             gr.Plot(PlaceHolder="Wav/Pause Plot", label="wav_pause_plot"),
             gr.Textbox(placeholder="Recording Feedback", label="message")],

    title="Laronix's Voice Quality Checking System Demo",
    description=description,
    allow_flagging="manual",
    flagging_dir="./exp/Kathryn_p326",
    flagging_options=["Perfect", "Susbicious_Speaking_Rate",
                      "Susbicious_Naturalness", "Susbicious_Pause"],
    examples=examples,
    css="body {background-color: green}"
)

print("Launch examples")

demo = gr.TabbedInterface([info, iface], tab_names=[
                          "Participant information", "Experiment"])

demo.launch(share=True, auth=[("Kathryn", "Laronix1130")])
