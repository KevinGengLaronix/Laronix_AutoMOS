import numpy as np
import librosa.display
import matplotlib.pyplot as plt

# Plot_UV
def plot_UV(signal, audio_interv, sr):
    fig, ax = plt.subplots(nrows=2, sharex=True)
    librosa.display.waveshow(signal, sr=sr, ax=ax[0])
    uv_flag = np.zeros(len(signal))
    for i in audio_interv:
        uv_flag[i[0]: i[1]] = 1

    ax[1].plot(np.arange(len(signal))/sr, uv_flag, "r")
    ax[1].set_ylim([-0.1,  1.1])
    return fig