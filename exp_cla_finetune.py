#!/usr/bin/python3 python

"""
Composer classification
Note density
Next sequence prediction

Elapsed time (beat)
"""

import os
from abc import ABC
from pathlib import Path
from copy import deepcopy
from typing import List, Dict, Any
from dataclasses import dataclass

from transformers import BertConfig, TrainingArguments, BertForSequenceClassification

from classes import Experiment, Baseline, DataConfig, TestingConfig, TokenizationConfig
from dataset import DatasetClaComposer, DatasetClaEmotion, DataCollatorClaFinetune
from constants import *


class BaselineClaFinetune(Baseline, ABC):
    def create_data_collator(self) -> DataCollatorClaFinetune:
        return DataCollatorClaFinetune(self.pad_token)  # by default, simply pad the input_ids

    def create_model(self):
        model = BertForSequenceClassification(self.model_config)
        return model


class BaselineComposer(BaselineClaFinetune):
    def create_dataset(self, files_paths: List[Path]):
        return DatasetClaComposer(files_paths, self.data_config.min_seq_len, self.data_config.max_seq_len,
                                  self.model_config.num_labels, self.bos_token, self.eos_token)


class BaselineEmotion(BaselineClaFinetune):
    def create_dataset(self, files_paths: List[Path]):
        return DatasetClaEmotion(files_paths, self.data_config.min_seq_len, self.data_config.max_seq_len,
                                 self.bos_token, self.eos_token)


@dataclass
class FineTuningTask:
    name: str
    dataset: str
    nb_classes: int
    baseline_cls: Any
    special_arguments: Dict[str, Dict[str, Any]] = None


nb_note_densities = NOTE_DENSITY_RANGE.stop - NOTE_DENSITY_RANGE.start
special_arguments_emotion = {
    "data": {"min_seq_len": MIN_SEQ_LEN_CLA_EMOTION, "max_seq_len": MAX_SEQ_LEN_CLA_EMOTION},
    "test": {"min_seq_len": MIN_SEQ_LEN_TEST_CLA_EMOTION, "max_seq_len": MAX_SEQ_LEN_TEST_CLA_EMOTION},
    "train": {"max_steps": TRAINING_STEPS_CLA_FT_EMOTION},
}
ftts = [
    FineTuningTask("composer_20", "GiantMIDI", 20, BaselineComposer),
    FineTuningTask("composer_100", "GiantMIDI", 100, BaselineComposer),
    FineTuningTask("emotion", "EMOPIA", 4, BaselineEmotion, special_arguments_emotion),
]
tokenizations = ['TSD', 'REMI', 'MIDILike', 'BPNO']  # for all fine-tunings

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
    per_device_train_batch_size=BATCH_SIZE_CLA_FT,
    per_device_eval_batch_size=BATCH_SIZE_TEST_CLA,
    gradient_accumulation_steps=GRAD_ACC_STEPS_CLA,
    eval_accumulation_steps=EVAL_ACCUMULATION_STEPS,
    eval_steps=VALID_INTVL,
    learning_rate=LEARNING_RATE_CLA_FT,
    weight_decay=WEIGHT_DECAY,
    max_grad_norm=GRADIENT_CLIP_NORM,
    max_steps=TRAINING_STEPS_CLA_FT,
    lr_scheduler_type=LR_SCHEDULER_CLA,
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
data_config = DataConfig(VALID_SPLIT, TEST_SPLIT, MIN_SEQ_LEN_CLA_FT, MAX_SEQ_LEN_CLA_FT)
test_config = TestingConfig(BATCH_SIZE_TEST_CLA, MIN_SEQ_LEN_TEST_CLA, MAX_SEQ_LEN_TEST_CLA)

experiments = []
for ftt in ftts:
    exp_name = f'cla_{ftt.name}_{ftt.dataset}'
    model_config.num_labels = ftt.nb_classes  # overrides depending on ft task
    baselines = []
    for tokenization in tokenizations:
        data_conf_, test_conf_, model_conf_, train_conf_ = \
            map(deepcopy, [data_config, test_config, model_config, finetune_config])
        if ftt.special_arguments is not None:
            for name, conf in [("data", data_conf_), ("test", test_conf_), ("model", model_conf_),
                               ("train", train_conf_)]:
                if name in ftt.special_arguments:
                    for attr, val in ftt.special_arguments[name].items():
                        setattr(conf, attr, val)
        tok_config = TokenizationConfig(tokenization, VOCAB_SIZE_BPE_CLA, TOKENIZER_PARAMS)
        baselines.append(ftt.baseline_cls(tokenization, exp_name, ftt.dataset, SEED, tok_config, model_conf_,
                                          train_conf_, data_conf_, test_conf_))

    experiments.append(Experiment(exp_name, baselines, ftt.dataset))


if __name__ == '__main__':
    import json

    from torch.distributed import get_world_size, get_rank
    from transformers.trainer_utils import get_last_checkpoint, set_seed
    from evaluate import load as load_metric
    from pandas import DataFrame

    from training import train_model, is_training_done, preprocess_logits, create_subsets

    metrics_names = ["f1", "accuracy"]
    try:
        metrics_func = {metric: load_metric(metric, num_process=get_world_size(), process_id=get_rank(),
                                            experiment_id="cla")
                        for metric in metrics_names}
    except RuntimeError:
        metrics_func = {metric: load_metric(metric, experiment_id="cla") for metric in metrics_names}
    metrics = {name: DataFrame(None, tokenizations, [exp_.name for exp_ in experiments]) for name in metrics_names}

    def compute_metrics_ft_composer(eval_pred):
        """Computes metrics for pretraining.
        Must use proprocess_logits function that converts logits to predictions (argmax or sampling).

        :param eval_pred: EvalPrediction containing predictions and labels
        :return: metrics
        """
        predictions, labels = eval_pred
        metric_res = {"accuracy": metrics_func["accuracy"].compute(predictions=predictions.flatten(),
                                                                   references=labels.flatten())["accuracy"],
                      "f1": metrics_func["f1"].compute(predictions=predictions.flatten(),
                                                       references=labels.flatten(),
                                                       average="micro")["f1"]}
        return metric_res

    for exp_ in experiments:
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
                model_ = model_.from_pretrained(get_last_checkpoint(pt_path),
                                                num_labels=baseline_.model_config.num_labels)

            # Fine-tune model and test it
            train_model(baseline_.training_config, model_, dataset_train, dataset_valid, dataset_test, collator,
                        compute_metrics=compute_metrics_ft_composer, preprocess_logits_for_metrics=preprocess_logits)

        # Read test results and write to dataframe
        for baseline_ in exp_.baselines:
            with open(baseline_.run_path / "test_results.json") as json_file:
                test_results = json.load(json_file)
            for metric in metrics_names:
                metrics[metric][exp_.name][baseline_.tokenization] = test_results[f"test_{metric}"]

    # Export to LaTeX tabular
    for metric_name, metric_df in metrics.items():
        metric_df.to_latex(Path("runs", f"cla_ft_test_metrics_latex_{metric_name}.text"))
