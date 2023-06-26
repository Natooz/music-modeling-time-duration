#!/usr/bin/python3 python

"""
Contrastive representation learning
"""

import os
from pathlib import Path
from copy import deepcopy
from typing import List

from transformers import BertConfig, TrainingArguments
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from classes import Experiment, Baseline, DataConfig, TestingConfig, TokenizationConfig
from dataset import DatasetMIDI, DataCollatorContrastive, DataCollatorContrastiveSupervised
from constants import *
from model_cl import BertForCL


class BaselineContrastive(Baseline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # divides the batch sizes by 2 as we will repeat the sequences
        self.training_config.per_device_train_batch_size = int(self.training_config.per_device_train_batch_size / 2)
        self.training_config.per_device_eval_batch_size = int(self.training_config.per_device_eval_batch_size / 2)

    def create_data_collator(self, **kwargs) -> DataCollatorContrastive:
        return DataCollatorContrastive(self.pad_token, self.bos_token, self.eos_token)

    def create_dataset(self, files_paths: List[Path], *args, **kwargs):
        additional_kwargs = kwargs.get("additional_kwargs", None)
        return DatasetMIDI(files_paths, self.data_config.min_seq_len - 2, self.data_config.max_seq_len - 2,
                           output_labels=True, additional_kwargs=additional_kwargs)

    def create_model(self):
        model = BertForCL(baseline_.model_config, POOLER_TYPE_CON, TEMPERATURE_CON)
        return model


class BaselineContrastiveSupervised(BaselineContrastive):
    def create_data_collator(self, **kwargs) -> DataCollatorContrastiveSupervised:
        return DataCollatorContrastiveSupervised(self.pad_token, self.bos_token, self.eos_token, self.tokenizer,
                                                 AUGMENTATIONS_TESTS_CON)


datasets = ["GiantMIDI"]
tokenizations = ["TSD", "MIDILike", "REMI", "BPNO"]  # for all fine-tunings
toks_fig_name = ["TS+Dur", "TS+NOff", "Pos+Dur", "Pos+NOff"]

model_config = BertConfig(
    vocab_size=None,
    num_labels=2,  # will be overridden / modified when creating baselines
    hidden_size=MODEL_DIM,
    num_hidden_layers=MODEL_NB_LAYERS,
    num_attention_heads=MODEL_NB_HEADS,
    intermediate_size=MODEL_D_FFWD,
    hidden_dropout_prob=DROPOUT,
    attention_probs_dropout_prob=DROPOUT,
    max_position_embeddings=MODEL_NB_POS_ENC_PARAMS,
    type_vocab_size=2,
)
finetune_config = TrainingArguments(
    "", False, True, True, False, "steps",
    per_device_train_batch_size=BATCH_SIZE_CON,
    per_device_eval_batch_size=BATCH_SIZE_TEST_CON,
    gradient_accumulation_steps=GRAD_ACC_STEPS_CON,
    eval_accumulation_steps=EVAL_ACCUMULATION_STEPS,
    eval_steps=VALID_INTVL,
    learning_rate=LEARNING_RATE_CON,
    weight_decay=WEIGHT_DECAY,
    max_grad_norm=GRADIENT_CLIP_NORM,
    max_steps=TRAINING_STEPS_CON,
    lr_scheduler_type=LR_SCHEDULER_CON,
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
)
data_config = DataConfig(VALID_SPLIT, TEST_SPLIT, MIN_SEQ_LEN_CON, MAX_SEQ_LEN_CON)
test_config = TestingConfig(BATCH_SIZE_TEST_CON, MIN_SEQ_LEN_TEST_CON, MAX_SEQ_LEN_TEST_CON)

experiments = []
for dataset in datasets:
    exp_name = f'contrastive_{dataset}'
    baselines = []
    for tokenization in tokenizations:
        tok_config = TokenizationConfig(tokenization, VOCAB_SIZE_BPE_CLA, TOKENIZER_PARAMS)
        baselines.append(BaselineContrastive(tokenization, exp_name, dataset, SEED, tok_config, deepcopy(model_config),
                                             deepcopy(finetune_config), data_config, test_config))
for dataset in datasets:
    exp_name = f'contrastive_sup_{dataset}'
    baselines = []
    for tokenization in tokenizations:
        tok_config = TokenizationConfig(tokenization, VOCAB_SIZE_BPE_CLA, TOKENIZER_PARAMS)
        baselines.append(
            BaselineContrastiveSupervised(tokenization, exp_name, dataset, SEED, tok_config, deepcopy(model_config),
                                          deepcopy(finetune_config), data_config, test_config))

    experiments.append(Experiment(exp_name, baselines, dataset))

metrics_names = ["isoscore", "lPCA", "MLE", "MOM", "MiND_ML", "TwoNN", "FisherS"]


def plot_ridge(df: pd.DataFrame, out_path: Path):
    # Theme
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0), 'axes.linewidth':2})
    pal = sns.cubehelix_palette(10, rot=-.25, light=.7)

    # create a grid with a row for each 'Language'
    g = sns.FacetGrid(df, row="g", hue="g", aspect=15, height=.5, palette=pal)

    # map df - Kernel Density Plot of IMDB Score for each Language
    g.map(sns.kdeplot, "x",
          bw_adjust=.5, clip_on=False,
          fill=True, alpha=1, linewidth=1.5)
    g.map(sns.kdeplot, "x", clip_on=False, color="w", lw=2, bw_adjust=.5)
    # passing color=None to refline() uses the hue mapping
    g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

    # Define and use a simple function to label the plot in axes coordinates
    def label(x, color, label):
        ax = plt.gca()
        ax.text(0, .2, label, fontweight="bold", color=color,
                ha="left", va="center", transform=ax.transAxes)

    # iterate grid to plot labels
    g.map(label, "x")

    # Set the subplots to overlap
    g.figure.subplots_adjust(hspace=-.25)

    # Remove axes details that don't play well with overlap
    g.set_titles("")
    # g.set(yticks=[], ylabel="", xlabel="cosine similarity")
    g.set(yticks=[], ylabel="", xlabel="")
    g.despine(bottom=True, left=True)
    plt.savefig(out_path, bbox_inches="tight")


if __name__ == '__main__':
    from typing import List, Tuple, Union, Dict
    import json

    from miditok import MIDITokenizer, TokSequence
    from torch import Tensor, LongTensor, isin, diagonal, mean, stack, no_grad, cat
    from torch.utils.data import DataLoader
    from torch.nn.utils.rnn import pad_sequence
    from transformers.trainer_utils import get_last_checkpoint, set_seed
    from tqdm import tqdm
    import skdim.id as skdim_id
    import numpy as np

    from training import train_model, is_training_done, create_subsets

    id_func = [getattr(skdim_id, id_func) for id_func in metrics_names[1:]]

    def compute_metrics_ft_composer(eval_pred):
        predictions, _ = eval_pred  # predictions are sequence representations
        metric_res = {}
        for func in id_func:
            id_ = func()
            try:
                id_ = id_.fit(predictions)
                metric_res[id_.__class__.__name__] = id_.dimension_
            except:
                pass

        # Augmentations
        return metric_res


    def get_possible_offsets(samples: LongTensor, tokenizer: MIDITokenizer, offsets: Dict) -> Dict:
        possible_offsets = {name: [] for name in offsets}

        # Decode BPE if needed
        if tokenizer.has_bpe:
            samples_no_bpe = []
            for sample in samples:
                seq = TokSequence(ids=sample.tolist(), ids_bpe_encoded=True)
                tokenizer.decode_bpe(seq)
                samples_no_bpe.append(LongTensor(seq.ids))
            # Need to pad as lengths have changed
            samples = pad_sequence(samples_no_bpe, True, tokenizer["PAD_None"])

        # Get min and max pitches
        pitch_ids_vocab = cat([LongTensor(tokenizer.token_ids_of_type("Pitch")),
                               LongTensor(tokenizer.token_ids_of_type("NoteOn"))])
        for si, sample in enumerate(samples):
            ids_pitch = sample[isin(sample, pitch_ids_vocab)]
            min_id, max_id = min(ids_pitch), max(ids_pitch)
            min_pitch = int(tokenizer[int(min_id)].split("_")[1])
            max_pitch = int(tokenizer[int(max_id)].split("_")[1])

            # Possible offsets
            for name, offset in offsets.items():
                if tokenizer.pitch_range.start <= min_pitch + offset[0] < tokenizer.pitch_range.stop and \
                        tokenizer.pitch_range.start <= max_pitch + offset[0] < tokenizer.pitch_range.stop:
                    possible_offsets[name].append(si)

        return possible_offsets

    def data_augmentation_tokens(
            tokens: Union[np.ndarray, List[int]],
            tokenizer,
            offsets: Tuple[int, int, int],
    ) -> List[int]:
        pitch_offset, vel_offset = offsets[:2]

        # Decode BPE
        bpe_decoded = False
        if tokenizer.has_bpe:
            in_seq = TokSequence(ids=tokens.tolist() if isinstance(tokens, np.ndarray) else tokens,
                                 ids_bpe_encoded=True)
            tokenizer.decode_bpe(in_seq)
            tokens = in_seq.ids
            bpe_decoded = True

        # Converts to np array if necessary
        if not isinstance(tokens, np.ndarray):
            tokens = np.array(tokens)
        augmented = tokens.copy()

        if pitch_offset != 0:
            # Get the maximum and lowest pitch in original track
            pitch_tokens = np.concatenate([np.array(tokenizer.token_ids_of_type("Pitch")),
                                           np.array(tokenizer.token_ids_of_type("NoteOn"))])
            mask_pitch = np.isin(tokens, pitch_tokens)

            # Perform augmentation on pitch
            augmented[mask_pitch] += pitch_offset

        # Velocity augmentation
        if vel_offset != 0:
            vel_tokens = np.array(tokenizer.token_ids_of_type("Velocity"))

            mask = np.isin(augmented, vel_tokens)

            augmented[mask] += vel_offset
            augmented[mask] = np.clip(augmented[mask], vel_tokens[0], vel_tokens[-1])

        # Convert array to list and reapply BPE if necessary
        seq = TokSequence(ids=augmented.tolist())
        if bpe_decoded:
            tokenizer.apply_bpe(seq)

        return seq.ids


    for exp_ in experiments:
        cosim_metrics_all = {name_: [] for name_ in AUGMENTATIONS_TESTS_CON}  # {aug_name: (baseline, numbers)}
        for baseline_ in exp_.baselines:
            # Check training is not already done and init
            if is_training_done(baseline_.run_path):
                continue
            last_checkpoint = get_last_checkpoint(str(baseline_.run_path)) if baseline_.run_path.exists() else None
            baseline_.training_config.resume_from_checkpoint = last_checkpoint  # only applies to resume from ft
            model_ = baseline_.create_model()

            # Load data
            set_seed(baseline_.seed)  # set before loading checkpoint
            token_paths = list(Path(baseline_.tokens_path).glob('**/*.json'))
            dataset_ = baseline_.create_dataset(files_paths=token_paths)
            dataset_train, dataset_valid, dataset_test = create_subsets(dataset_, [baseline_.data_config.valid_ratio,
                                                                                   baseline_.data_config.test_ratio])
            collator = baseline_.create_data_collator()

            # Load pretrained weights if necessary
            if last_checkpoint is None:  # no finetuning yet, we load weights from pretrained
                pt_path = Path('runs', 'cla_pre_trained', f'{exp_.dataset}_{baseline_.tokenization}')  # common to cla
                model_ = model_.from_pretrained(get_last_checkpoint(pt_path))

            # Fine-tune model and test it
            train_model(baseline_.training_config, model_, dataset_train, dataset_valid, dataset_test, collator,
                        compute_metrics=compute_metrics_ft_composer)

            # Test with data augmentations
            dataloader_test = DataLoader(dataset_test, batch_size=baseline_.training_config.eval_batch_size,
                                         collate_fn=collator)
            last_ft_checkpoint = get_last_checkpoint(baseline_.run_path)
            model_ = model_.from_pretrained(last_ft_checkpoint).to(model_.device)  # need to move back to GPU
            model_.sim.temp = 1
            model_.eval()
            cosim_metrics = {name_: [] for name_ in AUGMENTATIONS_TESTS_CON}
            with no_grad():
                for batch in tqdm(dataloader_test, desc=f"Testing ({baseline_.name}) with augmentation"):

                    # Get offsets that can be applied to all samples of the batch
                    valid_offsets = get_possible_offsets(batch["input_ids"], baseline_.tokenizer,
                                                         AUGMENTATIONS_TESTS_CON)

                    # Compute similarities
                    for name_, valid_samples in valid_offsets.items():
                        if len(valid_samples) == 0:
                            continue
                        shifted_samples = [
                            LongTensor(data_augmentation_tokens(batch["input_ids"][s].numpy(),
                                                                baseline_.tokenizer, AUGMENTATIONS_TESTS_CON[name_]))
                            for s in valid_samples
                        ]
                        input_ids = stack([batch["input_ids"][s] for s in valid_samples]).to(model_.device)
                        attention_mask = stack([batch["attention_mask"][s] for s in valid_samples]).to(model_.device)
                        input_ids2 = pad_sequence(shifted_samples, batch_first=True,
                                                  padding_value=baseline_.pad_token).to(model_.device)
                        attention_mask2 = (input_ids2 != baseline_.pad_token).int()
                        # res = model_(input_ids, input_ids2, attention_mask)  # (N,N) would need labels
                        z1 = model_(input_ids, attention_mask=attention_mask)  # (N,E)
                        z2 = model_(input_ids2, attention_mask=attention_mask2)  # (N,E)
                        sim = model_.sim(z1.pooler_output, z2.pooler_output)  # (N)
                        cosim_metrics[name_] += sim.tolist()

            # Compute mean, plot, print and save results
            cosim_metrics_means = {}
            for name_, values in cosim_metrics.items():
                cosim_metrics_means[name_] = float(mean(Tensor(values)))
                cosim_metrics_all[name_].append(values)

            print("DATA AUGMENTATION COSINE SIM RESULTS:")
            print(cosim_metrics_means)

            with open(baseline_.run_path / "data_aug_cosine_sim_results.json", "w") as fp:
                json.dump(cosim_metrics_means, fp)

        for aug_name, values in cosim_metrics_all.items():  # {aug_name: (baseline, numbers)}
            dists = []
            toks = []
            for i, res in enumerate(values):
                dists += res
                toks += len(res) * [toks_fig_name[i]]
            df_ = pd.DataFrame(dict(x=dists, g=toks))
            plot_ridge(df_, out_path=exp_.run_path / f"{aug_name}.pdf")
