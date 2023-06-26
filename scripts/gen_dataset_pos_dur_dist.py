#!/usr/bin/python3 python

from pathlib import Path

from miditok import TSD
from miditoolkit import MidiFile
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import colors

from constants import TOKENIZER_PARAMS


def gradientbars(bars, ydata, cmap):
    ax = bars[0].axes
    lim = ax.get_xlim()+ax.get_ylim()
    ax.axis(lim)
    for bar in bars:
        bar.set_facecolor("none")
        x, y = bar.get_xy()
        w, h_ = bar.get_width(), bar.get_height()
        grad = np.atleast_2d(np.linspace(0, 1*h_/max(ydata), 256)).T
        ax.imshow(grad, extent=[x, x+w, y, y+h_], origin='lower', aspect="auto",
                  norm=cm.colors.NoNorm(vmin=0, vmax=1), cmap=cmap)


if __name__ == '__main__':
    # Will simply split the maestro dataset files into train / valid / test subsets
    file_paths = list(Path("data", "POP909").glob("**/*.mid"))

    onsets = []
    offsets = []
    durations = []
    tokenizer = TSD(**TOKENIZER_PARAMS)
    time_division = max(tokenizer.beat_res.values())  # ticks per beat, here 1 pos = 1 tick
    ticks_per_bar = time_division * 4
    nb_positions = ticks_per_bar
    durations_tick = [tokenizer._token_duration_to_ticks(".".join([str(d) for d in dur]), time_division)
                      for dur in tokenizer.durations]
    for path in file_paths:
        midi = MidiFile(path)
        tokenizer.preprocess_midi(midi)
        for note in midi.instruments[0].notes:
            onsets.append(int((note.start / midi.ticks_per_beat) * time_division) % ticks_per_bar)
            offsets.append(int((note.end / midi.ticks_per_beat) * time_division) % ticks_per_bar)
            dur = int(((note.end - note.start) / midi.ticks_per_beat) * time_division)
            durations.append(min(dur, max(durations_tick)))
            # dur = note.end - note.start
            # index = np.argmin(np.abs(np.array(durations_tick) - dur))
            # durations.append(durations_tick[index])

    # Prepare vars
    bins = list(range(nb_positions + 1))
    ticks = list(range(0, len(bins) - 1, time_division)) + [bins[-2]]

    # Plot onsets
    plt.figure()
    h, _ = np.histogram(onsets, bins=bins, density=True)
    cmap = colors.LinearSegmentedColormap.from_list("", ["cornflowerblue", "deepskyblue", "lightskyblue"])
    gradientbars(plt.bar(range(len(bins) - 1), h, width=1, edgecolor='k'), h, cmap)
    plt.xticks(ticks, ticks, fontsize=18)
    plt.yticks(fontsize=12)
    plt.ylabel("Probability", fontsize=22)
    plt.xlabel("Position", fontsize=22)
    plt.savefig(Path("runs", "gen_POP909", "onsets_POP909.pdf"), bbox_inches="tight")
    plt.clf()

    # Plot offsets
    plt.figure()
    h, _ = np.histogram(offsets, bins=bins, density=True)
    cmap = colors.LinearSegmentedColormap.from_list("", ["slateblue", "royalblue", "cornflowerblue"])
    gradientbars(plt.bar(range(len(bins) - 1), h, width=1, edgecolor='k'), h, cmap)
    plt.xticks(ticks, ticks, fontsize=18)
    plt.yticks(fontsize=12)
    plt.ylabel("Probability", fontsize=22)
    plt.xlabel("Position", fontsize=22)
    plt.savefig(Path("runs", "gen_POP909", "offsets_POP909.pdf"), bbox_inches="tight")
    plt.clf()

    # Plot durations
    durations_xticks = ([7, 12, 14, 16, 17, 18, 19, 20], [1, 2, 3, 4, 5, 6, 7, 8])
    plt.figure()
    h, _ = np.histogram(durations, bins=durations_tick, density=True)
    cmap = colors.LinearSegmentedColormap.from_list("", ["darkslateblue", "mediumslateblue", "slateblue"])
    gradientbars(plt.bar(range(len(durations_tick) - 1), h, width=1, edgecolor='k'), h, cmap)
    plt.xticks(*durations_xticks, fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylabel("Probability", fontsize=22)
    plt.xlabel("Beat", fontsize=22)
    plt.savefig(Path("runs", "gen_POP909", "durations_POP909.pdf"), bbox_inches="tight")
    plt.clf()
    """tokenizations = len(durations) * ["POP909"]
    df = pd.DataFrame({"x": durations, "tokenization": tokenizations})
    sns.catplot(
        data=df,
        x="tokenization", y="x", kind="boxen",
    )"""
