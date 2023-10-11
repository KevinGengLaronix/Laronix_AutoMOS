from pathlib import Path
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
import sys
import pdb

threshold = 0.3
if __name__ == "__main__":
    wer_csv = sys.argv[1] 
    df = pd.read_csv(wer_csv)
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(25, 15))

    # Hist for distribution
    ax[0].set_xlabel("Word Error Rate")
    ax[0].set_ylabel("Counts")
    ax[0].set_xlim(left=0.0, right=df['wer'].max())
    ax[0].hist(df['wer'], bins=50)
    ax[0].axvline(x=threshold, color="r")
    # plt.savefig("hist.png")
    
    # Line curve for each sentences
    colors = ['green' if x < threshold else 'red' for x in df['wer']]

    new_ids = [str(x).split('.')[0] for x in df['id']]    
    ax[1].set_xlabel("IDs")
    ax[1].set_ylabel("Word Error Rate")
    ax[1].scatter(new_ids, df['wer'], c=colors, marker='o')
    ax[1].vlines(new_ids, ymin=0, ymax=df['wer'], colors='grey', linestyle='dotted', label='Vertical Lines')
    ax[1].axhline(y=threshold, xmin=0, xmax=len(new_ids), color='r')

    # ax[0].axhline(y=threshold, color="black")

    # for i, v in enumerate(df['wer']):
        # plt.text(str(df['id'][i]).split('.')[0], -2, str(df['id'][i]), ha='center', fontsize=3)

    ax[1].set_xticklabels(new_ids, rotation=90, fontsize=10)
    ax[1].tick_params(axis='x', width=20)
    # ax[1].set_xlim(10, len(df['id']) + 10)
    plt.tight_layout()
    pdb.set_trace()
    # fig.savefig("%s/%s.png"%(Path(sys.argv[1]).parent, sys.argv[1].split('/')[-1]), format='png')
    fig.savefig("%s.png"%(sys.argv[1]), format='png')
