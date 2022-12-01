
from random import sample
from sys import flags
import gradio as gr
import torchaudio
import torch
import torch.nn as nn
import lightning_module
import pdb
import jiwer
import numpy as np

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
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")
phoneme_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")
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
        indices = (torch.arange(new_length) * (self.input_rate / self.output_rate))
        round_down = wav[:, indices.long()]
        round_up = wav[:, (indices.long() + 1).clamp(max=wav.size(-1) - 1)]
        output = round_down * (1. - indices.fmod(1.)).unsqueeze(0) + round_up * indices.fmod(1.).unsqueeze(0)
        return output

model = lightning_module.BaselineLightningModule.load_from_checkpoint("epoch=3-step=7459.ckpt").eval()

def calc_mos(audio_path, ref):
    wav, sr = torchaudio.load(audio_path)
    osr = 16_000
    batch = wav.unsqueeze(0).repeat(10, 1, 1)
    csr = ChangeSampleRate(sr, osr)
    out_wavs = csr(wav)
    # ASR
    trans = p(audio_path)["text"]
    # WER
    wer = jiwer.wer(ref, trans, truth_transform=transformation, hypothesis_transform=transformation)
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

description ="""
MOS prediction demo using UTMOS-strong w/o phoneme encoder model, which is trained on the main track dataset.
This demo only accepts .wav format. Best at 16 kHz sampling rate.

Paper is available [here](https://arxiv.org/abs/2204.02152)

Add ASR based on wav2vec-960, currently only English available.
Add WER interface.
""" 
## Auto load examples

refs = np.loadtxt("Arthur_the_rat.txt", delimiter="\n", dtype="str")
refs_ids = [x.split(" ")[0] for x in refs]
refs_txt = [" ".join(x.split(" ")[1:]) for x in refs]
from pathlib import Path
ref_wavs =[str(x) for x in sorted(Path("./Patient_sil_trim_16k_normed_5_snr_40/Arthur").glob("**/*.wav"))]

referance_id = gr.Textbox(value="ID",
                    placeholder="Utter ID",
                    label="Reference_ID")
referance_textbox = gr.Textbox(value="",
                    placeholder="Input reference here",
                    label="Reference")
pdb.set_trace()
## Set up interface

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            ref_wav = gr.Audio(source="upload", type="filepath", label="Pneumatic_audio")
            referance_textbox = gr.Textbox(value="Once upon a time there was a young rat named Author who couldn’t make up his mind.",
                    placeholder="Input reference here",
                    label="Reference")
            ref_botton = gr.Button(value="Calculate reference TEP feature")
        with gr.Column():
            ref_P_MOS = gr.Textbox(placeholder="Predicted MOS", label="Predicted MOS")
            ref_hypothesis = gr.Textbox(placeholder="Hypothesis", label="Hypothesis")
            ref_WER = gr.Textbox(placeholder="Word Error Rate", label = "WER")
            ref_Predicted_Phonemes = gr.Textbox(placeholder="Predicted Phonemes", label="Predicted Phonemes")
            ref_PPM = gr.Textbox(placeholder="Phonemes per minutes", label="PPM")
        ref_botton.click(fn=calc_mos, 
                         inputs=[ref_wav, referance_textbox], 
                         outputs=[ref_P_MOS,
                                  ref_hypothesis,
                                  ref_WER,
                                  ref_Predicted_Phonemes,
                                  ref_PPM])    
    with gr.Row(scale=20):
        multi_input =  [[x, y] for x, y in zip(ref_wavs, refs_txt)]
        examples = gr.Examples(examples=multi_input, inputs=[ref_wav, referance_textbox])

demo.launch(debug=True)
            
# iface = gr.Interface(
#   fn=calc_mos,
#   inputs=[gr.Audio(source="microphone", type='filepath', label="Audio_to_evaluate"),
#           referance_textbox],
#   outputs=[gr.Textbox(placeholder="Predicted MOS", label="Predicted MOS"),
#            gr.Textbox(placeholder="Hypothesis", label="Hypothesis"),
#            gr.Textbox(placeholder="Word Error Rate", label = "WER"),
#            gr.Textbox(placeholder="Predicted Phonemes", label="Predicted Phonemes"),
#            gr.Textbox(placeholder="Phonemes per minutes", label="PPM")],
#   title="Laronix's Voice Quality Checking System Demo",
#   description=description,
#   allow_flagging="manual",
#   examples=refs_txt,
# )
# iface.launch(debug=True)