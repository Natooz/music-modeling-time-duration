#!/usr/bin/python3 python

"""
Pre-train models for other tasks
"""

import os
from pathlib import Path
from copy import deepcopy

from torch import argmax
from torch.distributed import get_world_size, get_rank
from transformers import BertForPreTraining, TrainingArguments
from evaluate import load as load_metric

from classes import DataConfig
from exp_cla_finetune import experiments
from constants import *


pretrain_config = TrainingArguments(
    "", False, True, True, False, "steps",
    per_device_train_batch_size=BATCH_SIZE_PT,
    per_device_eval_batch_size=BATCH_SIZE_TEST_CLA,
    gradient_accumulation_steps=GRAD_ACC_STEPS_PT,
    eval_accumulation_steps=EVAL_ACCUMULATION_STEPS,
    eval_steps=VALID_INTVL,
    learning_rate=LEARNING_RATE_PT,
    weight_decay=WEIGHT_DECAY,
    max_grad_norm=GRADIENT_CLIP_NORM,
    max_steps=TRAINING_STEPS_PT,
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
data_config_pt = DataConfig(VALID_SPLIT, 0, MIN_SEQ_LEN_PT, MAX_SEQ_LEN_PT)
metrics_names = ["accuracy", "f1"]
try:
    metrics = {metric: load_metric(metric, num_process=get_world_size(), process_id=get_rank(), experiment_id="pt")
               for metric in metrics_names}
except RuntimeError:
    metrics = {metric: load_metric(metric, experiment_id="pt") for metric in metrics_names}


def preprocess_logits(logits, _):
    """Preprocesses the logits before accumulating them during evaluation.
    This allows to significantly reduce the memory usage and make the training tractable.
    """
    preds = (argmax(logits[0], dim=-1), argmax(logits[1], dim=-1))  # long dtype
    return preds


def compute_metrics_pt(eval_pred):
    """Computes metrics for pretraining.
    Must use proprocess_logits function that converts logits to predictions (argmax or sampling).

    :param eval_pred: EvalPrediction containing predictions and labels
    :return: metrics
    """
    (predictions_mlm, predictions_nsp), (labels_mlm, labels_nsp) = eval_pred
    pad_mask = labels_mlm != -100
    labels_mlm, predictions = labels_mlm[pad_mask], predictions_mlm[pad_mask]
    metric_res = {"accuracy_mlm": metrics["accuracy"].compute(predictions=predictions.flatten(),
                                                              references=labels_mlm.flatten())["accuracy"],
                  "f1_nsp": metrics["f1"].compute(predictions=predictions_nsp.flatten(),
                                                  references=labels_nsp.flatten(),
                                                  average="micro")["f1"]}
    return metric_res


if __name__ == '__main__':
    from transformers.trainer_utils import set_seed, get_last_checkpoint

    from dataset import DatasetMIDI, DataCollatorClaPreTrain
    from training import train_model, split_object, is_training_done

    for exp_ in experiments:
        for baseline_ in exp_.baselines:
            # Adjust data and training config for pretraining
            pt_path = Path('runs', 'cla_pre_trained', f'{exp_.dataset}_{baseline_.tokenization}')
            baseline_.data_config = deepcopy(data_config_pt)
            baseline_.training_config = deepcopy(pretrain_config)
            baseline_.training_config.output_dir = str(pt_path)
            baseline_.training_config.logging_dir = str(pt_path)
            if exp_.dataset == "EMOPIA":
                baseline_.data_config.min_seq_len = MIN_SEQ_LEN_CLA_EMOTION
                baseline_.data_config.max_seq_len = MAX_SEQ_LEN_CLA_EMOTION
                baseline_.training_config.max_steps = TRAINING_STEPS_PT_EMOTION

            # pre-trained weights are common to all subsequent tasks
            if is_training_done(baseline_.run_path):
                continue
            last_checkpoint = get_last_checkpoint(str(baseline_.run_path)) if baseline_.run_path.exists() else None
            baseline_.training_config.resume_from_checkpoint = last_checkpoint
            model_ = BertForPreTraining(baseline_.model_config)

            # Load data
            set_seed(baseline_.seed)
            token_paths = list(Path(baseline_.tokens_path).glob('**/*.json'))
            paths_train, paths_valid, paths_test = split_object(token_paths, [baseline_.data_config.valid_ratio,
                                                                              baseline_.data_config.test_ratio])
            dataset_train = DatasetMIDI(paths_train,
                                        baseline_.data_config.min_seq_len - 3,
                                        baseline_.data_config.max_seq_len - 3)
            dataset_valid = DatasetMIDI(paths_valid,
                                        baseline_.data_config.min_seq_len - 3,
                                        baseline_.data_config.max_seq_len - 3)
            collator = DataCollatorClaPreTrain(pad_token=baseline_.pad_token,
                                               bos_token=baseline_.bos_token,
                                               eos_token=baseline_.eos_token,
                                               mask_token=baseline_.mask_token,
                                               sep_token=baseline_.sep_token,
                                               vocab_size=len(baseline_.tokenizer),
                                               special_tokens=baseline_.special_tokens,
                                               mlm_probability=MASK_RATIO_CLA_PT)

            # Pre-train the model
            train_model(baseline_.training_config, model_, dataset_train, dataset_valid, data_collator=collator,
                        compute_metrics=compute_metrics_pt, preprocess_logits_for_metrics=preprocess_logits)
