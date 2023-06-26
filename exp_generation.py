#!/usr/bin/python3 python

"""
Lists the Experiment baselines and training
"""

import os
from typing import List, Tuple, Optional, Union
from pathlib import Path
from copy import deepcopy

from transformers import GPT2LMHeadModel, GPT2Config, Seq2SeqTrainingArguments, GenerationConfig
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
import torch
from matplotlib import cm
from matplotlib import colors

from classes import Experiment, Baseline, DataConfig, TestingConfig, TokenizationConfig
from dataset import DatasetMIDI, DataCollatorStandard
from constants import *


class Model(GPT2LMHeadModel):
    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            pad_on_left: Optional[bool] = None,  # Subclassing to change signature for data collator
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        return super().forward(input_ids, past_key_values, attention_mask, token_type_ids, position_ids, head_mask,
                               inputs_embeds, encoder_hidden_states, encoder_attention_mask, labels, use_cache,
                               output_attentions, output_hidden_states, return_dict)


class BaselineGen(Baseline):
    def create_dataset(self, files_paths: List[Path], *args, **kwargs):
        pad_on_left = kwargs.get("pad_on_left", False)
        return DatasetMIDI(
            files_paths,
            self.data_config.min_seq_len - 1,
            self.data_config.max_seq_len - 1,
            self.tokenizer,
            True,
            pad_on_left,
        )

    def create_data_collator(self, pad_on_left: bool = False, shift_labels: bool = False) -> DataCollatorStandard:
        return DataCollatorStandard(self.pad_token, self.bos_token, pad_on_left=pad_on_left, shift_labels=shift_labels)

    def create_model(self):
        self.model_config.vocab_size = len(self.tokenizer)
        model = Model(self.model_config)
        model.generation_config = self.generation_config
        return model


model_config = GPT2Config(
    vocab_size=None,
    n_positions=MODEL_NB_POS_ENC_PARAMS,
    n_embd=MODEL_DIM,
    n_layer=MODEL_NB_LAYERS,
    n_head=MODEL_NB_HEADS,
    n_inner=MODEL_D_FFWD,
    resid_pdrop=DROPOUT,
    embd_pdrop=DROPOUT,
    attn_pdrop=DROPOUT,
    use_cache=False,  # not compatible with gradient checkpointing, prevents warnings during training
)
training_config = Seq2SeqTrainingArguments(
    "", False, True, True, False, "steps",
    per_device_train_batch_size=BATCH_SIZE_GEN,
    per_device_eval_batch_size=BATCH_SIZE_TEST_GEN,
    gradient_accumulation_steps=GRAD_ACC_STEPS_GEN,
    eval_accumulation_steps=EVAL_ACCUMULATION_STEPS,
    eval_steps=VALID_INTVL,
    learning_rate=LEARNING_RATE_GEN,
    weight_decay=WEIGHT_DECAY,
    max_grad_norm=GRADIENT_CLIP_NORM,
    max_steps=TRAINING_STEPS_GEN,
    lr_scheduler_type=LR_SCHEDULER_GEN,
    warmup_ratio=WARMUP_RATIO,
    log_level="debug",
    logging_strategy="steps",
    logging_steps=LOG_STEPS_INTVL,
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    save_total_limit=SAVE_TOTAL_LIMIT,
    no_cuda=not USE_CUDA,
    seed=SEED,
    fp16=USE_AMP,
    local_rank=int(os.getenv("LOCAL_RANK", -1)),  # for DDP
    load_best_model_at_end=True,
    label_smoothing_factor=LABEL_SMOOTHING,
    optim="adamw_torch",
    report_to=["tensorboard"],  # logging_dir will be set within Baseline class
    ddp_find_unused_parameters=DDP_FIND_UNUSED_PARAMETERS,
    ddp_bucket_cap_mb=DDP_BUCKET_CAP_MB,
    gradient_checkpointing=USE_GRADIENT_CHECKPOINTING,
    full_determinism=True,
    use_mps_device=False,
    predict_with_generate=True,
)
data_config = DataConfig(VALID_SPLIT, TEST_SPLIT, MIN_SEQ_LEN_GEN, MAX_SEQ_LEN_GEN)
test_config = TestingConfig(
    batch_size=BATCH_SIZE_TEST_GEN,
    min_seq_len=MIN_SEQ_LEN_TEST_GEN,
    max_seq_len=MAX_SEQ_LEN_TEST_GEN,
)
generation_config = GenerationConfig(
    max_length=None,
    max_new_tokens=NB_INFERENCES_GEN,
    num_beams=NUM_BEAMS,
    temperature=TEMPERATURE_SAMPLING,
    top_k=TOP_K,
    top_p=TOP_P,
    epsilon_cutoff=EPSILON_CUTOFF,
    eta_cutoff=ETA_CUTOFF,
)

datasets = ["POP909"]
tokenizations = ["TSD", "REMI", "MIDILike", "BPNO"]
experiments = []
for dataset in datasets:
    exp_name = f"gen_{dataset}"
    baselines = []
    for tokenization in tokenizations:
        data_conf_, test_conf_, model_conf_, train_conf_, gen_conf_ = \
            map(deepcopy, [data_config, test_config, model_config, training_config, generation_config])

        tok_config = TokenizationConfig(tokenization, VOCAB_SIZE_BPE_GEN, TOKENIZER_PARAMS)
        baselines.append(BaselineGen(tokenization, exp_name, dataset, SEED, tok_config, model_conf_, train_conf_,
                                     data_conf_, test_conf_, gen_conf_))

    experiments.append(Experiment(exp_name, baselines, dataset))


def save_generation_tokens(prompt: List[int], generated: List[int], tokenizer, out_dir: Path,
                           file_name: Union[int, str]):
    r"""Saves generated tokens, as json and MIDi files.
    :param prompt: original sample (prompt) used for the generation.
    :param generated: generated sequence
    :param tokenizer: tokenizer object.
    :param out_dir: output directory.
    :param file_name: file name, with no extension (.json and .mid will be added).
    """
    tokens = [generated, prompt, prompt + generated]
    midi = tokenizer.tokens_to_midi(deepcopy(tokens), time_division=TIME_DIVISION)  # copy as inplace decompose bpe op
    midi.instruments[0].name = f'Continuation of original sample ({len(generated)} tokens)'
    midi.instruments[1].name = f'Original sample ({len(prompt)} tokens)'
    midi.instruments[2].name = f'Original sample and continuation'
    midi.dump(out_dir / f'{file_name}.mid')
    tokenizer.save_tokens(tokens, out_dir / f'{file_name}.json')


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
    from miditok import MIDITokenizer, TokSequence
    from miditoolkit import MidiFile
    from functools import partial
    import numpy as np
    from transformers.trainer_utils import set_seed, get_last_checkpoint
    from matplotlib import pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from tqdm import tqdm

    from training import train_model, split_object, is_training_done
    from metrics import tse


    def compute_metrics(eval_pred, tokenizer: MIDITokenizer, out_dir_: Path):
        """Computes metrics for pretraining.
        Must use proprocess_logits function that converts logits to predictions (argmax or sampling).

        :param eval_pred: EvalPrediction containing predictions and labels
        :param tokenizer:
        :param out_dir_:
        :return: metrics
        """
        predictions, labels = eval_pred

        tse_ = []
        for i, (lab, pred) in enumerate(zip(labels, predictions)):
            # Preprocess tokens
            lab = lab[lab != -100]
            pred = pred[pred != 0]  # remove padding
            pred = pred[len(lab):].tolist()  # but prompt
            save_generation_tokens(lab.tolist(), pred, tokenizer, out_dir_, i)
            tse_.append(list(tse(pred, tokenizer)))

        tse_ = np.array(tse_)
        metric_res = {
            "tse_type": float(np.mean(tse_[:, 0])),
            "tse_time": float(np.mean(tse_[:, 1])),
            "tse_ndup": float(np.mean(tse_[:, 2])),
            "tse_nnon": float(np.mean(tse_[:, 3])),
            "tse_nnof": float(np.mean(tse_[:, 4])),
        }

        return metric_res


    for exp_ in experiments:

        # Split data here, so that we use the exact same test files for all baselines
        # Doing so allows fair human evaluation of the same conditional / prompted generation
        # We assume they have the same data_config
        set_seed(exp_.baselines[0].seed)
        files_names = [p.relative_to(exp_.baselines[0].tokens_path)
                       for p in exp_.baselines[0].tokens_path.glob('**/*.json')]
        names_train, names_valid, names_test = split_object(files_names, [exp_.baselines[0].data_config.valid_ratio,
                                                                          exp_.baselines[0].data_config.test_ratio])

        for baseline_ in exp_.baselines:
            if is_training_done(baseline_.run_path):
                continue
            last_checkpoint = get_last_checkpoint(str(baseline_.run_path)) if baseline_.run_path.exists() else None
            baseline_.training_config.resume_from_checkpoint = last_checkpoint
            model_ = baseline_.create_model()
            (baseline_.run_path / "gen").mkdir(parents=True, exist_ok=True)

            # Load data
            paths_train = [baseline_.tokens_path / name for name in names_train]
            paths_valid = [baseline_.tokens_path / name for name in names_valid]
            paths_test = [baseline_.tokens_path / name for name in names_test]
            dataset_train = baseline_.create_dataset(files_paths=paths_train)
            dataset_valid = baseline_.create_dataset(files_paths=paths_valid, pad_on_left=True)
            dataset_test = baseline_.create_dataset(files_paths=paths_test, pad_on_left=True)
            collator = baseline_.create_data_collator()

            # Train model if not already done
            comp_metric = partial(compute_metrics, tokenizer=baseline_.tokenizer, out_dir_=baseline_.run_path / "gen")
            train_model(baseline_.training_config, model_, dataset_train, dataset_valid, dataset_test,
                        data_collator=collator, compute_metrics=comp_metric)

        # Gather features
        time_division = max(exp_.baselines[0].tokenizer.beat_res.values())  # ticks per beat, here 1 pos = 1 tick
        ticks_per_bar = time_division * 4
        nb_positions = ticks_per_bar
        bins = list(range(nb_positions + 1))
        ticks = list(range(0, len(bins) - 1, time_division)) + [bins[-2]]
        durations_tick = [
            exp_.baselines[0].tokenizer._token_duration_to_ticks(".".join([str(d) for d in dur]), time_division)
            for dur in exp_.baselines[0].tokenizer.durations
        ]
        for baseline_ in tqdm(exp_.baselines, desc=f"Analyzing gen features ({exp_.name})"):
            gen_files_paths = (baseline_.run_path / "gen").glob("**/*.mid")
            gen_token_files_paths = (baseline_.run_path / "gen").glob("**/*.json")

            # Plots next token type matrices
            token_types = {tok: i for i, tok in enumerate(baseline_.tokenizer.tokens_types_graph.keys())}
            for special_token in baseline_.tokenizer.special_tokens:
                del token_types[special_token]
            token_successions = np.zeros((len(token_types), len(token_types)))  # (N,N), (first_tok, next_tok)
            for token_file_path in gen_token_files_paths:
                tokens = baseline_.tokenizer.load_tokens(token_file_path)["ids"][0]
                tok_seq = TokSequence(ids=tokens, ids_bpe_encoded=True)
                baseline_.tokenizer.decode_bpe(tok_seq)
                for i, token in enumerate(tok_seq.tokens):
                    if i == len(tok_seq) - 1:
                        continue
                    next_type = tok_seq.tokens[i + 1].split("_")[0]
                    token_successions[token_types[token.split("_")[0]], token_types[next_type]] += 1
            for i in range(len(token_successions)):
                if (total := np.sum(token_successions[i])) != 0:
                    token_successions[i] /= total
            plt.figure(figsize=(3, 3))
            ax = plt.gca()
            im = ax.imshow(token_successions)
            plt.yticks(list(range(len(token_types))), list(token_types.keys()), size=10)
            plt.xticks(list(range(len(token_types))), list(token_types.keys()), size=10)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=-40, ha="left", rotation_mode="anchor")
            plt.ylabel("")
            plt.xlabel("Next token")
            ax.xaxis.set_label_position('top')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
            plt.savefig(Path(exp_.run_path, f"next_token_types_{baseline_.tokenization}.pdf"), bbox_inches="tight")
            plt.clf()

            # Plots positions and durations
            onsets = []
            offsets = []
            durations = []
            for file_path in gen_files_paths:
                midi = MidiFile(file_path)
                for note in midi.instruments[0].notes:
                    onsets.append(int((note.start / midi.ticks_per_beat) * time_division) % ticks_per_bar)
                    offsets.append(int((note.end / midi.ticks_per_beat) * time_division) % ticks_per_bar)
                    durations.append(int(((note.end - note.start) / midi.ticks_per_beat) * time_division))

            # Plot onsets
            plt.figure()
            h, _ = np.histogram(onsets, bins=bins, density=True)
            cmap = colors.LinearSegmentedColormap.from_list("", ["cornflowerblue", "deepskyblue", "lightskyblue"])
            gradientbars(plt.bar(range(len(bins) - 1), h, width=1, edgecolor='k'), h, cmap)
            plt.xticks(ticks, ticks, fontsize=18)
            plt.yticks(fontsize=12)
            plt.ylabel("Probability", fontsize=22)
            plt.xlabel("Position", fontsize=22)
            plt.savefig(Path(exp_.run_path, f"onsets_{baseline_.tokenization}.pdf"), bbox_inches="tight")
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
            plt.savefig(Path(exp_.run_path, f"offsets_{baseline_.tokenization}.pdf"), bbox_inches="tight")
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
            plt.savefig(Path(exp_.run_path, f"durations_{baseline_.tokenization}.pdf"), bbox_inches="tight")
            plt.clf()
