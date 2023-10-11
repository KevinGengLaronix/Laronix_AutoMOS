"""
TODO:
    + [x] Load Configuration
    + [ ] Checking
    + [ ] Better saving directory
"""
import numpy as np
from pathlib import Path
import jiwer
import pdb
import torch.nn as nn
import torch
import torchaudio
import gradio as gr
from logging import PlaceHolder
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import yaml
from transformers import pipeline
import librosa
import librosa.display
import matplotlib.pyplot as plt


# local import
import sys

sys.path.append("src")
import lightning_module

# Load automos
config_yaml = sys.argv[1]
with open(config_yaml, "r") as f:
    # pdb.set_trace()
    try:
        config = yaml.safe_load(f)
    except FileExistsError:
        print("Config file Loading Error")
        exit()

# Auto load examples
refs = np.loadtxt(config["ref_txt"], delimiter="\n", dtype="str")
refs_ids = [x.split()[0] for x in refs]
refs_txt = [" ".join(x.split()[1:]) for x in refs]
ref_feature = np.loadtxt(config["ref_feature"], delimiter=",", dtype="str")
ref_wavs = [str(x) for x in sorted(Path(config["ref_wavs"]).glob("**/*.wav"))]

dummy_wavs = [None for x in np.arange(len(ref_wavs))]

refs_ppm = np.array(ref_feature[:, -1][1:], dtype="str")

reference_id = gr.Textbox(value="ID", placeholder="Utter ID", label="Reference_ID")

reference_textbox = gr.Textbox(
    value="Input reference here",
    placeholder="Input reference here",
    label="Reference",
)
reference_PPM = gr.Textbox(placeholder="Pneumatic Voice's PPM", label="Ref PPM")

# Set up interface
# remove dummpy wavs, ue the same ref_wavs for eval wavs
print("Preparing Examples")
examples = [
    [w, w_, i, x, y] for w, w_, i, x, y in zip(ref_wavs, ref_wavs, refs_ids, refs_txt, refs_ppm)
]

p = pipeline(
    "automatic-speech-recognition",
    model="KevinGeng/whipser_medium_en_PAL300_step25",
    device=0,
)

# WER part
transformation = jiwer.Compose(
    [
        jiwer.RemovePunctuation(),
        jiwer.ToLowerCase(),
        jiwer.RemoveWhiteSpace(replace_by_space=True),
        jiwer.RemoveMultipleSpaces(),
        jiwer.ReduceToListOfListOfWords(word_delimiter=" "),
    ]
)

# WPM part
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")
phoneme_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")


class ChangeSampleRate(nn.Module):
    def __init__(self, input_rate: int, output_rate: int):
        super().__init__()
        self.output_rate = output_rate
        self.input_rate = input_rate

    def forward(self, wav: torch.tensor) -> torch.tensor:
        # Only accepts 1-channel waveform input
        wav = wav.view(wav.size(0), -1)
        new_length = wav.size(-1) * self.output_rate // self.input_rate
        indices = torch.arange(new_length) * (self.input_rate / self.output_rate)
        round_down = wav[:, indices.long()]
        round_up = wav[:, (indices.long() + 1).clamp(max=wav.size(-1) - 1)]
        output = round_down * (1.0 - indices.fmod(1.0)).unsqueeze(0) + (
            round_up * indices.fmod(1.0).unsqueeze(0)
        )
        return output


# MOS model
model = lightning_module.BaselineLightningModule.load_from_checkpoint(
    "src/epoch=3-step=7459.ckpt"
).eval()

# Get Speech Interval

def get_speech_interval(signal, db):
    audio_interv = librosa.effects.split(signal, top_db=db)
    pause_end = [x[0] for x in audio_interv[1:]]
    pause_start = [x[1] for x in audio_interv[0:-1]]
    pause_interv = [[x, y] for x, y in zip(pause_start, pause_end)]
    return audio_interv, pause_interv

# plot UV


def plot_UV(signal, audio_interv, sr):
    fig, ax = plt.subplots(nrows=2, sharex=True)
    librosa.display.waveshow(signal, sr=sr, ax=ax[0])
    uv_flag = np.zeros(len(signal))
    for i in audio_interv:
        uv_flag[i[0] : i[1]] = 1

    ax[1].plot(np.arange(len(signal)) / sr, uv_flag, "r")
    ax[1].set_ylim([-0.1, 1.1])
    return fig

def calc_mos(_, audio_path, id, ref, pre_ppm, fig=None):
    if audio_path == None:
        audio_path = _
        print("using ref audio as eval audio since it's empty")

    wav, sr = torchaudio.load(audio_path)
    if wav.shape[0] != 1:
        wav = wav[0, :]
    print(wav.shape)

    osr = 16000
    batch = wav.unsqueeze(0).repeat(10, 1, 1)
    csr = ChangeSampleRate(sr, osr)
    out_wavs = csr(wav)

    # ASR
    trans = jiwer.ToLowerCase()(p(audio_path)["text"])

    # WER
    wer = jiwer.wer(
        ref,
        trans,
        truth_transform=transformation,
        hypothesis_transform=transformation,
    )
    # MOS
    batch = {
        "wav": out_wavs,
        "domains": torch.tensor([0]),
        "judge_id": torch.tensor([288]),
    }
    with torch.no_grad():
        output = model(batch)
    predic_mos = output.mean(dim=1).squeeze().detach().numpy() * 2 + 3

    # Phonemes per minute (PPM)
    with torch.no_grad():
        logits = phoneme_model(out_wavs).logits
    phone_predicted_ids = torch.argmax(logits, dim=-1)
    phone_transcription = processor.batch_decode(phone_predicted_ids)
    lst_phonemes = phone_transcription[0].split(" ")

    # VAD for pause detection
    wav_vad = torchaudio.functional.vad(wav, sample_rate=sr)
    # pdb.set_trace()
    a_h, p_h = get_speech_interval(wav_vad.numpy(), db=40)
    # print(a_h)
    # print(len(a_h))
    fig_h = plot_UV(wav_vad.numpy().squeeze(), a_h, sr=sr)
    ppm = len(lst_phonemes) / (wav_vad.shape[-1] / sr) * 60

    error_msg = "!!! ERROR MESSAGE !!!\n"
    if audio_path == _ or audio_path == None:
        error_msg += "ERROR: Fail recording, Please start from the beginning again."
        return (
            fig_h,
            predic_mos,
            trans,
            wer,
            phone_transcription,
            ppm,
            error_msg,
        )
    if ppm >= float(pre_ppm) + float(config["thre"]["maxppm"]):
        error_msg += "ERROR: Please speak slower.\n"
    elif ppm <= float(pre_ppm) - float(config["thre"]["minppm"]):
        error_msg += "ERROR: Please speak faster.\n"
    elif predic_mos <= float(config["thre"]["AUTOMOS"]):
        error_msg += "ERROR: Naturalness is too low, Please try again.\n"
    elif wer >= float(config["thre"]["WER"]):
        error_msg += "ERROR: Intelligibility is too low, Please try again\n"
    else:
        error_msg = (
            "GOOD JOB! Please 【Save the Recording】.\nYou can start recording the next sample."
        )

    return (
        fig_h,
        predic_mos,
        trans,
        wer,
        phone_transcription,
        ppm,
        error_msg,
    )

with open("src/description.html", "r", encoding="utf-8") as f:
    description = f.read()
# description

refs_ppm = np.array(ref_feature[:, -1][1:], dtype="str")

reference_id = gr.Textbox(value="ID", placeholder="Utter ID", label="Reference_ID", visible=False)
reference_textbox = gr.Textbox(
    value="Input reference here",
    placeholder="Input reference here",
    label="Reference",
)
reference_PPM = gr.Textbox(placeholder="Pneumatic Voice's PPM", label="Ref PPM", visible=False)

# Flagging setup

# Interface
# Participant Information
def record_part_info(name, gender, first_lng):
    message = "Participant information is successfully collected."
    id_str = "%s_%s_%s" % (name, gender[0], first_lng[0])

    if name == None:
        message = "ERROR: Name Information incomplete!"
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
        message = "ERROR: Please select one english proficiency only"
        id_str = "ERROR"

    return message, id_str


# information page not using now
name = gr.Textbox(placeholder="Name", label="Name")
gender = gr.CheckboxGroup(["Male", "Female"], label="gender")
first_lng = gr.CheckboxGroup(
    [
        "B1 Intermediate",
        "B2: Upper Intermediate",
        "C1: Advanced",
        "C2: Proficient",
    ],
    label="English Proficiency (CEFR)",
)

msg = gr.Textbox(placeholder="Evaluation for valid participant", label="message")
id_str = gr.Textbox(placeholder="participant id", label="participant_id")

info = gr.Interface(
    fn=record_part_info,
    inputs=[name, gender, first_lng],
    outputs=[msg, id_str],
    title="Participant Information Page",
    allow_flagging="never",
    css="body {background-color: blue}",
)
# Experiment
if config["exp_id"] == None:
    config["exp_id"] = Path(config_yaml).stem

## This is the theme for the interface
css = """
.ref_text textarea {font-size: 40px !important}
.message textarea {font-size: 40px !important}
"""

my_theme = gr.themes.Default().set(
    button_primary_background_fill="#75DA99",
    button_primary_background_fill_dark="#DEF2D7",
    button_primary_text_color="black",
    button_secondary_text_color="black",
)

# Callback for saving the recording
callback = gr.CSVLogger()

with gr.Blocks(css=css, theme=my_theme) as demo:
    with gr.Column():
        with gr.Row():
            ref_audio = gr.Audio(
                source="microphone",
                type="filepath",
                label="Reference_Audio",
                container=True,
                interactive=False,
                visible=False,
            )
            with gr.Row():
                eval_audio = gr.Audio(
                    source="microphone",
                    type="filepath",
                    container=True,
                    label="Audio_to_Evaluate",
                )
                b_redo = gr.ClearButton(
                    value="Redo", variant="stop", components=[eval_audio], size="sm"
                )
                reference_textbox = gr.Textbox(
                    value="Input reference here",
                    placeholder="Input reference here",
                    label="Reference",
                    interactive=True,
                    elem_classes="ref_text",
                )
                with gr.Accordion("Input for Development", open=False):
                    reference_id = gr.Textbox(
                        value="ID",
                        placeholder="Utter ID",
                        label="Reference_ID",
                        visible=True,
                    )
                    reference_PPM = gr.Textbox(
                        placeholder="Pneumatic Voice's PPM",
                        label="Ref PPM",
                        visible=True,
                    )
        with gr.Row():
            b = gr.Button(value="1.Submit", variant="primary", elem_classes="submit")

            # TODO
            # b_more = gr.Button(value="Show More", elem_classes="verbose")
        with gr.Row():
            inputs = [
                ref_audio,
                eval_audio,
                reference_id,
                reference_textbox,
                reference_PPM,
            ]
            e = gr.Examples(examples, inputs, examples_per_page=5)

    with gr.Column():
        with gr.Row():
            ## output block
            msg = gr.Textbox(
                placeholder="Recording Feedback",
                label="Message",
                interactive=False,
                elem_classes="message",
            )
            with gr.Accordion("Output for Development", open=False):
                wav_plot = gr.Plot(PlaceHolder="Wav/Pause Plot", label="wav_pause_plot", visible=True)

                predict_mos = gr.Textbox(
                    placeholder="Predicted MOS",
                    label="Predicted MOS",
                    visible=True,
                )

                hyp = gr.Textbox(placeholder="Hypothesis", label="Hypothesis", visible=True)

                wer = gr.Textbox(placeholder="Word Error Rate", label="WER", visible=True)

                predict_pho = gr.Textbox(
                    placeholder="Predicted Phonemes",
                    label="Predicted Phonemes",
                    visible=True,
                )

                ppm = gr.Textbox(
                    placeholder="Phonemes per minutes",
                    label="PPM",
                    visible=True,
                )
            outputs = [
                wav_plot,
                predict_mos,
                hyp,
                wer,
                predict_pho,
                ppm,
                msg,
            ]

            # b = gr.Button("Submit")
            b.click(fn=calc_mos, inputs=inputs, outputs=outputs, api_name="Submit")

        # Logger
        callback.setup(
            components=[
                eval_audio,
                reference_id,
                reference_textbox,
                reference_PPM,
                predict_mos,
                hyp,
                wer,
                ppm,
                msg],
            flagging_dir="./exp/%s" % config["exp_id"],
        )

        with gr.Row():
            b2 = gr.Button("2. Save the Recording", variant="primary", elem_id="save")
            js_confirmed_saving = "(x) => confirm('Recording Saved!')"
            # eval_audio,
            b2.click(
                lambda *args: callback.flag(args),
                inputs=[
                    eval_audio,
                    reference_id,
                    reference_textbox,
                    reference_PPM,
                    predict_mos,
                    hyp,
                    wer,
                    ppm,
                    msg,
                ],
                outputs=None,
                preprocess=False,
                api_name="flagging",
            )
        with gr.Row():
            b3 = gr.ClearButton(
                [
                    ref_audio,
                    eval_audio,
                    reference_id,
                    reference_textbox,
                    reference_PPM,
                    predict_mos,
                    hyp,
                    wer,
                    ppm,
                    msg,
                ],
                value="3.Clear All",
                elem_id="clear",
            )

demo.launch(share=True)