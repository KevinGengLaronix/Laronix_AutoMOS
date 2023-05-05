import numpy as np
import librosa.display
import matplotlib.pyplot as plt

# Plot_UV

def plot_UV(signal, audio_interv, sr):
    fig, ax = plt.subplots(nrows=2, sharex=True)
    librosa.display.waveshow(signal, sr=sr, ax=ax[0])
    ax[0].set_title("Signal")
    ax[1].set_title("U/V")
    uv_flag = np.zeros(len(signal))
    for i in audio_interv:
        uv_flag[i[0]: i[1]] = 1

    ax[1].plot(np.arange(len(signal))/sr, uv_flag, "r")
    ax[1].set_ylim([-0.1,  1.1])
    return fig

# Get Speech Interval


def get_speech_interval(signal, db):
    audio_interv = librosa.effects.split(signal, top_db=db)
    pause_end = [x[0] for x in audio_interv[1:]]
    pause_start = [x[1] for x in audio_interv[0: -1]]
    pause_interv = [[x, y] for x, y in zip(pause_start, pause_end)]
    return audio_interv, pause_interv
