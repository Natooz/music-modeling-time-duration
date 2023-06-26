#!/usr/bin/python3 python

""" Script to transform the MIDI datasets into tokens.

"""
from pathlib import Path
from random import choices

from miditoolkit import MidiFile
from transformers.trainer_utils import set_seed

from constants import BPE_NB_FILES_LIM, DATA_AUGMENTATION_OFFSETS


def is_midi_valid(midi: MidiFile, min_nb_tracks: int = 1, four_beats_per_bar_only: bool = False) -> bool:
    """ Returns whether a MIDI file is valid or not
    The conditions are:
        - contains the minimum number of beats given
        - contains the minimum number of tracks given
        - 4/* time signature only if four_beats_per_bar_only is True
    :param midi: MIDI object to valid
    :param min_nb_tracks: number min of tracks (default 1 to pass everything)
    :param four_beats_per_bar_only: will discard MIDIs with time signatures other than 4/*
    :return: True if the MIDI is valid, else False
    """
    if len(midi.instruments) < min_nb_tracks:
        return False
    if four_beats_per_bar_only and any(ts.numerator != 4 for ts in midi.time_signature_changes):
        return False

    return True


if __name__ == "__main__":
    from exp_cla_finetune import experiments as exp_cla
    from exp_contrastive import experiments as exp_con
    from exp_generation import experiments as exp_gen

    for exp in exp_cla + exp_con + exp_gen:
        for baseline in exp.baselines:
            if (baseline.tokens_path / "config.txt").is_file():
                continue

            # If not already done, tokenize MIDI dataset without BPE + perform data augmentation
            tokens_path_no_bpe = Path("data", f"{exp.dataset}_{baseline.tokenization}_noBPE")
            if not tokens_path_no_bpe.exists():
                midi_paths = list(exp.data_path_midi.glob("**/*.mid"))
                baseline.tokenizer.tokenize_midi_dataset(midi_paths, tokens_path_no_bpe, is_midi_valid,
                                                         DATA_AUGMENTATION_OFFSETS)

            # Learn and apply BPE on dataset
            set_seed(42)  # for file lim random selection
            tokens_paths = list(tokens_path_no_bpe.glob("**/*.json"))
            if len(tokens_paths) > BPE_NB_FILES_LIM:
                tokens_paths = choices(tokens_paths, k=BPE_NB_FILES_LIM)
            baseline.tokens_path.mkdir(exist_ok=True, parents=True)
            baseline.tokenizer.learn_bpe(baseline.tokenization_config.vocab_size_bpe, tokens_paths=tokens_paths)
            baseline.tokenizer.apply_bpe_to_dataset(tokens_path_no_bpe, baseline.tokens_path)
            baseline.tokenizer.save_params(baseline.tokens_path / "config.txt")
