# TODO:
+ [ ] Design directory structure
+ [ ] Usage


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