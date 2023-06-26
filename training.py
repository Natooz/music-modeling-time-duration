"""
Training functions
"""

from typing import List, Any, Callable, Union
from pathlib import Path

from torch import Tensor, randperm, cumsum, argmax, device
from torch.nn import Module
from torch.cuda import is_available as cuda_available
from torch.backends.mps import is_available as mps_available
from torch.utils.data import Dataset, Subset, random_split
from transformers import Trainer, Seq2SeqTrainer, TrainingArguments, Seq2SeqTrainingArguments, DataCollator, \
    TrainerCallback


def split_object(obj_, split_ratio: List[float]) -> List[Any]:
    r"""Create subsets of a dataset following split ratios.
    if sum(split_ratio) != 1, the remaining portion will be inserted as the first subset

    :param obj_: Object to split, must support indexing and implement the __len__ magic method.
    :param split_ratio: split ratios as a list of float
    :return: the list of subsets
    """
    assert all(0 <= ratio <= 1. for ratio in split_ratio), 'The split ratios must be comprise within [0,1]'
    assert sum(split_ratio) <= 1., 'The sum of split ratios must be inferior or equal to 1'
    nb_samples = len(obj_)
    len_subsets = [int(nb_samples * ratio) for ratio in split_ratio]
    if sum(split_ratio) != 1.:
        len_subsets.insert(0, nb_samples - sum(len_subsets))
    cum_sums = cumsum(Tensor(len_subsets), 0).long()
    idx = randperm(nb_samples)
    idx = [idx[offset - length: offset] for offset, length in zip(cum_sums, len_subsets)]
    split = [[obj_[idx__] for idx__ in idx_] for idx_ in idx]
    return split


def create_subsets(dataset: Dataset, split_ratio: List[float]) -> List[Subset]:
    r"""Create subsets of a dataset following split ratios.
    if sum(split_ratio) != 1, the remaining portion will be inserted as the first subset

    :param dataset: Dataset object, must implement the __len__ magic method.
    :param split_ratio: split ratios as a list of float
    :return: the list of subsets
    """
    assert all(0 <= ratio <= 1. for ratio in split_ratio), 'The split ratios must be comprise within [0,1]'
    assert sum(split_ratio) <= 1., 'The sum of split ratios must be inferior or equal to 1'
    len_subsets = [int(len(dataset) * ratio) for ratio in split_ratio]
    if sum(split_ratio) != 1.:
        len_subsets.insert(0, len(dataset) - sum(len_subsets))
    subsets = random_split(dataset, len_subsets)
    return subsets


def select_device(use_cuda: bool = True, use_mps: bool = True, log: bool = False) -> device:
    r"""Select the device on which PyTorch will run

    :param use_cuda: will run on nvidia GPU if available. (default: True)
    :param use_mps: will run on MPS device if available. (default: True)
    :param log: will log a warning message if a CUDA device is detected but not used. (default: False)
    :return: 'cpu' or 'cuda:0' device object.
    """
    if cuda_available():
        if use_cuda:
            return device("cuda:0")
        elif log:
            print("WARNING: You have a CUDA device, you should probably run with it")
    if mps_available():
        if use_mps:
            return device("mps")
        elif log:
            print("WARNING: You have a MPS device, you should probably run with it")
    return device('cpu')


def is_training_done(run_path: Path) -> bool:
    """Tells if a model has already been trained in the run_path directory,

    :param run_path: model training directory
    :return: if model has already been fully trained
    """
    if run_path.exists():
        if (run_path / "all_results.json").is_file():
            return True

    return False


def preprocess_logits(logits: Tensor, _: Tensor) -> Tensor:
    """Preprocesses the logits before accumulating them during evaluation.
    This allows to significantly reduce the memory usage and make the training tractable.
    """
    pred_ids = argmax(logits, dim=-1)  # long dtype
    return pred_ids


def train_model(
        training_config: Union[TrainingArguments, Seq2SeqTrainingArguments],
        model: Module,
        dataset_train: Dataset,
        dataset_valid: Dataset,
        dataset_test: Dataset = None,
        data_collator: DataCollator = None,
        compute_metrics: Callable = None,
        callbacks: List[TrainerCallback] = None,
        preprocess_logits_for_metrics: Callable = None,
):
    r"""Trains a model, given a baseline

    :param training_config: training configuration.
    :param model: model to trained.
    :param dataset_train: dataset for training data.
    :param dataset_valid: dataset for validation data.
    :param dataset_test: dataset for test / inference data.
    :param data_collator: collate function to use with dataloaders. (default: None)
    :param compute_metrics: function calling metrics. (default: None)
    :param preprocess_logits_for_metrics: function that preprocess logits before accumulating them during
            evaluation. (default: None)
    :param callbacks: list of TrainerCallback. (default: None)
    """
    # Init
    Path(training_config.output_dir).mkdir(parents=True, exist_ok=True)
    trainer = (Seq2SeqTrainer if isinstance(training_config, Seq2SeqTrainingArguments) else Trainer)(
        model=model,
        args=training_config,
        data_collator=data_collator,
        train_dataset=dataset_train,
        eval_dataset=dataset_valid,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    # Training
    train_result = trainer.train(resume_from_checkpoint=training_config.resume_from_checkpoint)
    trainer.save_model()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    # Inference / Test
    if dataset_test is not None:
        if isinstance(trainer, Seq2SeqTrainer) and trainer.args.predict_with_generate:
            trainer.preprocess_logits_for_metrics = None
        test_results = trainer.predict(dataset_test)
        trainer.log_metrics("test", test_results.metrics)
        trainer.save_metrics("test", test_results.metrics)
