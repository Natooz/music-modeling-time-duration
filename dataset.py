from pathlib import Path
from typing import Dict, Union, List, Tuple, Any, Iterable
from abc import ABC
from dataclasses import dataclass
from copy import deepcopy
from random import choice
import json
import csv

from torch import Tensor, LongTensor, cat, stack, full, arange
import torch
from torch.utils.data import Dataset, IterableDataset
import torchaudio
from miditok import MIDITokenizer, TokSequence
from miditoolkit import MidiFile, Instrument, TempoChange
from tqdm import tqdm


class DatasetABC(Dataset, ABC):
    def __init__(self, samples=None):
        self.samples = samples if samples is not None else []

    def __len__(self) -> int: return len(self.samples)

    def __repr__(self): return self.__str__()

    def __str__(self) -> str: return 'No data loaded' if len(self) == 0 else f'{len(self.samples)} samples'


class DatasetMIDI(DatasetABC):
    r"""Basic Dataset loading MIDI files.

    :param files_paths: list of paths to files to load.
    :param min_seq_len: minimum sequence length (in nb of tokens)
    :param max_seq_len: maximum sequence length (in nb of tokens)
    :param tokenizer: tokenizer object, to use to load MIDIs instead of tokens. (default: None)
    :param output_labels: will output a "labels" entry in the return item. (default: False)
    """

    def __init__(
            self,
            files_paths: List[Path],
            min_seq_len: int,
            max_seq_len: int,
            tokenizer: MIDITokenizer = None,
            output_labels: bool = False,
            pad_on_left: bool = None,
            additional_kwargs: Dict = None,
    ):
        self.output_labels = output_labels
        self.pad_on_left = pad_on_left
        self.additional_kwargs = additional_kwargs
        samples = []

        for file_path in tqdm(files_paths, desc=f"Loading data: {files_paths[0].parent}"):
            if file_path.suffix in ["mid", "midi", "MID", "MIDI"]:
                midi = MidiFile(file_path)
                for _ in range(len(midi.instruments) - 1):
                    del midi.instruments[1]  # removes all tracks except first one
                tokens = tokenizer.midi_to_tokens(midi)[0]
            else:
                with open(file_path) as json_file:
                    tokens = json.load(json_file)["ids"][0]  # first track
            if len(tokens) < min_seq_len:
                continue  # tokens of this file not long enough

            # Cut tokens in samples of appropriate length
            i = 0
            while i < len(tokens):
                if i >= len(tokens) - min_seq_len:
                    break  # last sample is too short
                samples.append(LongTensor(tokens[i:i + max_seq_len]))
                i += len(samples[-1])  # could be replaced with max_seq_len

        super().__init__(samples)

    def __getitem__(self, idx) -> Dict[str, LongTensor]:
        item = {"input_ids": self.samples[idx]}
        if self.output_labels:
            item["labels"] = self.samples[idx]
        if self.pad_on_left is not None:
            item["pad_on_left"] = self.pad_on_left
        if self.additional_kwargs is not None:
            for key, val in self.additional_kwargs:
                item[key] = val
        return item


class DatasetClaComposer(DatasetABC):
    r"""Dataset for composer classification.
    Only for GiantMIDI
    NSP: https://github.com/huggingface/transformers/blob/main/src/transformers/data/datasets/language_modeling.py
    For NSP, would need to add a SEP token to tokenizer vocab, and either mix sequences within Dataset
    (not dynamic) or override DataCollatorForLanguageModeling to shuffle sequences (dynamic).

    :param files_paths: list of paths to files to load.
    :param min_seq_len: minimum sequence length (in nb of tokens)
    :param max_seq_len: maximum sequence length (in nb of tokens)
    :param tokenizer: tokenizer object, to use to load MIDIs instead of tokens. (default: None)
    """

    def __init__(
            self,
            files_paths: List[Path],
            min_seq_len: int,
            max_seq_len: int,
            max_nb_composers: int,
            tokenizer: MIDITokenizer = None
    ):
        self.samples_composer_idx = []
        composers = {}  # stores composer_name: [samples_idx]

        for file_path in tqdm(files_paths, desc=f'Preparing data: {files_paths[0].parent}'):
            # Check file is good
            parts = file_path.name.split(', ')
            if len(parts) < 4:
                continue

            # Load tokens
            if file_path.suffix in ["mid", "midi", "MID", "MIDI"]:
                midi = MidiFile(file_path)
                for _ in range(len(midi.instruments) - 1):
                    del midi.instruments[1]  # removes all tracks except first one
                tokens = tokenizer.midi_to_tokens(midi)[0]
            else:
                with open(file_path) as json_file:
                    tokens = json.load(json_file)['ids'][0]  # first track
            if len(tokens) < min_seq_len:
                continue  # tokens of this file not long enough

            # Get composer name and store it if not already done
            composer = f'{parts[0]} {parts[1]}'
            if composer not in composers:
                composers[composer] = []

            # Cut tokens in samples of appropriate length
            i = 0
            while i < len(tokens):
                if i >= len(tokens) - min_seq_len:
                    break  # last sample is too short
                sample = LongTensor(tokens[i:i + max_seq_len])
                composers[composer].append(sample)
                i += max_seq_len

        '''# Remove composers < min_nb_samples
        composers_list = list(self.composers.keys())  # all of them, as a list
        for composer in composers_list:
            if len(self.composers[composer]) < min_nb_samples_per_composer:
                del self.composers[composer]'''

        # Keep only max_nb_composers, ones with most samples
        samples = []
        composers_sorted_per_nb_samples = sorted(composers.keys(), key=lambda x: len(composers[x]))
        self.nb_composers = 0
        for i, composer in enumerate(composers_sorted_per_nb_samples[-max_nb_composers:]):
            samples += composers[composer]
            self.samples_composer_idx += len(composers[composer]) * [i]
            self.nb_composers += 1

        del composers
        super().__init__(samples)

    def __getitem__(self, idx) -> Dict[str, Union[LongTensor, int]]:
        return {"input_ids": self.samples[idx],
                "labels": self.samples_composer_idx[idx]}


class DatasetClaEmotion(DatasetABC):
    r"""Dataset for emotion classification, with the EMOPIA dataset.

    :param files_paths: list of paths to files to load.
    :param min_seq_len: minimum sequence length (in nb of tokens)
    :param max_seq_len: maximum sequence length (in nb of tokens)
    :param tokenizer: tokenizer object, to use to load MIDIs instead of tokens. (default: None)
    """

    def __init__(
            self,
            files_paths: List[Path],
            min_seq_len: int,
            max_seq_len: int,
            tokenizer: MIDITokenizer = None
    ):
        samples = []
        samples_labels = []

        for file_path in tqdm(files_paths, desc=f'Preparing data: {files_paths[0].parent}'):
            # Load tokens
            if file_path.suffix in ["mid", "midi", "MID", "MIDI"]:
                midi = MidiFile(file_path)
                for _ in range(len(midi.instruments) - 1):
                    del midi.instruments[1]  # removes all tracks except first one
                tokens = tokenizer.midi_to_tokens(midi)[0]
            else:
                with open(file_path) as json_file:
                    tokens = json.load(json_file)['ids'][0]  # first track
            if len(tokens) < min_seq_len:
                continue  # tokens of this file not long enough

            # Get emotion label
            label = int(file_path.name[1]) - 1  # Q1/Q2/Q3/Q4

            # Cut tokens in samples of appropriate length
            i = 0
            while i < len(tokens):
                if i >= len(tokens) - min_seq_len:
                    break  # last sample is too short
                samples.append(LongTensor(tokens[i:i + max_seq_len]))
                samples_labels.append(label)
                i += max_seq_len

        self.samples_labels = LongTensor(samples_labels)
        super().__init__(samples)

    def __getitem__(self, idx) -> Dict[str, Union[LongTensor, int]]:
        return {"input_ids": self.samples[idx], "labels": self.samples_labels[idx]}


@dataclass
class DatasetMusicTranscriptionMaestro(IterableDataset):
    r"""Dataset class for the Maestro dataset for music transcription.

    :param subset_path: path to the subset of
    :param seq_len_enc: sequence length for the encoder, in nb of mel spectrogram frames.
    :param min_seq_len_dec: minimum sequence length for the decoder, in nb of symbolic tokens.
    :param max_seq_len_dec: maximum sequence length for the decoder, in nb of symbolic tokens.
    :param tokenizer: tokenizer object, to use to load MIDIs instead of tokens. (default: None)
    :param sample_rate: sample rate of audio.
    :param n_ftt: number of Fast Fourier Transformer (FTT / STFT) performed.
    :param hop_width: width / length of the sliding windows of STFT.
    :param n_mels: number of mel bins.
    :param safe_log_eps: epsilon value "flooring" log values == 0 before performing log.
    """

    subset_path: Path
    seq_len_enc: int  # fixed size len, no padding is performed (cant batch otherwise)
    min_seq_len_dec: int
    max_seq_len_dec: int
    tokenizer: MIDITokenizer
    sample_rate: int
    n_ftt: int
    win_length: int
    hop_width: int
    n_mels: int
    safe_log_eps: float
    device: torch.device
    pad_on_left: bool = False
    decoder_inputs: bool = True

    def __post_init__(self):
        self.min_seq_len_dec -= 2
        self.max_seq_len_dec -= 2
        self.transform = torchaudio.transforms.MelSpectrogram(
            self.sample_rate,
            self.n_ftt,
            win_length=self.win_length,
            hop_length=self.hop_width,
            n_mels=self.n_mels
        ).to(self.device)

    @torch.no_grad()
    def compute_log_mel(self, waveform: Tensor) -> Tensor:
        mel_specgram = self.transform(waveform.to(self.device)).t()  # (F,M)  for n_mels and nb_frames
        mel_specgram[mel_specgram < self.safe_log_eps] = self.safe_log_eps
        log_mel = torch.log(mel_specgram)
        return log_mel  # (F,M)

    def data_stream(self) -> Dict[str, Tensor]:
        with open(self.subset_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            next(csv_reader)  # skips first line
            for row in csv_reader:
                midi_path = Path("data", "Maestro", row[4])
                audio_path = Path("data", "Maestro", row[5])

                # Load audio, mix channels and resample it
                waveform, original_freq = torchaudio.load(audio_path, normalize=True)  # (2,T)
                if len(waveform.shape) > 1:
                    waveform = torch.sum(waveform, dim=0)  # mix channels (T)
                waveform = torchaudio.functional.resample(waveform, original_freq, self.sample_rate)

                # Compute mel spectrogram + log
                log_mel = self.compute_log_mel(waveform)

                # Split mels by chunks (batch items)
                chunks_log_mel = list(log_mel.split(self.seq_len_enc))  # [N(T,M)] N=nb chunks, T seq len
                if len(chunks_log_mel) > 0 and chunks_log_mel[-1].size(0) != self.seq_len_enc:
                    del chunks_log_mel[-1]
                if len(chunks_log_mel) == 0:
                    continue

                # Load MIDI and split notes by chunks
                midi = MidiFile(midi_path)
                tick_to_sec = torch.Tensor(midi.get_tick_to_time_mapping())
                frames_per_second = self.sample_rate / self.hop_width
                times_chunks = torch.arange(len(chunks_log_mel)) * self.seq_len_enc / frames_per_second
                chunk_duration = float(times_chunks[1])
                chunk_duration_tick = int(torch.argmin(torch.abs(tick_to_sec - chunk_duration)))
                chunks_notes = [[] for _ in range(len(chunks_log_mel))]
                for note in midi.instruments[0].notes:
                    chunk_id = int(tick_to_sec[note.start] // chunk_duration)  # more accurate than doing it in ticks
                    if chunk_id < len(chunks_log_mel):  # as last log_mel chunk has probably been deleted
                        note2 = deepcopy(note)
                        note2.start %= chunk_duration_tick
                        note2.end %= chunk_duration_tick
                        chunks_notes[chunk_id].append(note2)  # note times shifted

                # Convert notes to tokens
                def del_chunk_id(id_: int):
                    del chunks_log_mel[id_]
                    del chunks_notes[id_]

                chunks_tokens = []
                for i in range(len(chunks_log_mel) - 1, -1, -1):
                    if len(chunks_notes[i]) == 0:
                        del_chunk_id(i)
                        continue
                    midi_chunk = MidiFile(ticks_per_beat=midi.ticks_per_beat)
                    midi_chunk.instruments.append(Instrument(0))
                    midi_chunk.instruments[0].notes = chunks_notes[i]
                    midi_chunk.tempo_changes = [TempoChange(120, 0)]  # just for mocking
                    tokens = self.tokenizer(midi_chunk)[0]
                    if not self.min_seq_len_dec <= len(tokens) <= self.max_seq_len_dec:
                        del_chunk_id(i)
                        continue
                    chunks_tokens.append(tokens.ids)
                    i += 1

                # Finally return batch items, one after another
                for log_mels, tokens_ids in zip(chunks_log_mel, chunks_tokens):
                    batch_item = {"inputs_embeds": log_mels,
                                  "decoder_input_ids": LongTensor(tokens_ids),
                                  "pad_on_left": self.pad_on_left,
                                  "decoder_inputs": self.decoder_inputs}
                    yield batch_item

    def __iter__(self) -> Iterable:
        return iter(self.data_stream())

    def __getitem__(self, index):
        pass


class DataCollatorStandard:
    def __init__(
            self,
            pad_token: int,
            bos_token: int = None,
            eos_token: int = None,
            pad_on_left: bool = False,
            shift_labels: bool = False,
            labels_pad_idx: int = -100,
            add_bos_eos_to_labels: bool = False,
            inputs_kwarg_name: str = "input_ids",
            labels_kwarg_name: str = "labels",
    ):
        """Multifunction data collator, that can pad the sequences (right or left), add BOS and EOS tokens.
        Input_ids will be padded with the pad token given, while labels will be padded with -100.

        :param pad_token: PAD token
        :param bos_token: BOS token (default: None).
        :param eos_token: EOS token (default: None).
        :param pad_on_left: will pad sequence on the left (default: False).
        :param shift_labels: will shift inputs and labels for autoregressive training / teacher forcing.
        :param labels_pad_idx: padding idx for labels (default: -100).
        :param add_bos_eos_to_labels: will add BOS and/or EOS tokens to the labels (default: False).
        :param inputs_kwarg_name: name of dict / kwarg key for inputs (default: "input_ids").
        :param inputs_kwarg_name: name of dict / kwarg key for inputs (default: "labels_").
        """
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_on_left = pad_on_left
        self.shift_labels = shift_labels
        self.labels_pad_idx = labels_pad_idx
        self.add_bos_eos_to_labels = add_bos_eos_to_labels
        self.inputs_kwarg_name = inputs_kwarg_name
        self.labels_kwarg_name = labels_kwarg_name

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, LongTensor]:
        out_batch = {}
        pad_on_left = batch[0]["pad_on_left"] if "pad_on_left" in batch[0] else self.pad_on_left

        # Add BOS and EOS tokens + PAD to inputs
        x = None
        if self.inputs_kwarg_name in batch[0]:
            _add_bos_eos_tokens_to_batch(batch, self.inputs_kwarg_name, bos_tok=self.bos_token, eos_tok=self.eos_token)
            x = _pad_batch(batch, self.pad_token, self.inputs_kwarg_name, pad_on_left)

        # Add BOS and EOS tokens + PAD labels
        y = None
        if self.labels_kwarg_name in batch[0]:
            _add_bos_eos_tokens_to_batch(batch, self.labels_kwarg_name, bos_tok=self.bos_token, eos_tok=self.eos_token)
            y = _pad_batch(batch, self.labels_pad_idx, self.labels_kwarg_name, pad_on_left)

        # Shift labels
        if self.shift_labels:  # otherwise it's handled in models such as GPT2LMHead
            if x is not None:
                inputs = x
            elif y is not None:
                inputs = y
            else:
                raise ValueError("Either inputs or labels have to be specified by the Dataset.")
            x = inputs[:-1]
            y = inputs[1:]

        # Add inputs / labels to output batch
        if x is not None:
            out_batch[self.inputs_kwarg_name] = x
        if y is not None:
            out_batch[self.labels_kwarg_name] = y

        # Create attention mask (just for padding, causality is handled in models)
        attention_mask = (x != self.pad_token).int()
        if attention_mask.dim() == 3:
            attention_mask = attention_mask[..., 0]  # (N,T,Z) --> (N,T)
        out_batch["attention_mask"] = attention_mask

        return out_batch


class DataCollatorClaPreTrain:
    """
    Data collator used for language modeling. Inputs are dynamically padded to the maximum length of a batch if they
    are not all the same length.
    Inspired from transformers.DataCollatorForLanguageModeling
    """

    def __init__(
            self,
            pad_token: int,
            bos_token: int,
            eos_token: int,
            mask_token: int,
            sep_token: int,
            vocab_size: int,
            special_tokens: List[int],
            mlm_probability: float = 0.15,
            nsp_probability: float = 0.5,
            sentence_b_ratio: float = 0.5,
    ):
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.mask_token = mask_token
        self.sep_token = LongTensor([sep_token])
        self.vocab_size = vocab_size
        self.special_tokens = LongTensor(special_tokens)
        self.mlm_probability = mlm_probability
        self.nsp_probability = nsp_probability
        self.sentence_b_ratio = sentence_b_ratio

    def __call__(self, batch: List[Dict[str, Any]], return_tensors=None) -> Dict[str, Any]:
        _add_bos_eos_tokens_to_batch(batch, bos_tok=self.bos_token, eos_tok=self.eos_token)
        # Determine idx to mix for NSP
        batch, token_type_ids, next_sentence_label = _nsp_swap(batch,
                                                               self.nsp_probability,
                                                               self.sentence_b_ratio,
                                                               self.sep_token,
                                                               self.pad_token)

        # Pad and mask them
        masked_inputs, original_input = self.torch_mask_tokens(_pad_batch(batch, self.pad_token))
        attention_mask = (masked_inputs != self.pad_token).int()

        # If special token mask has been preprocessed, pop it from the dict.
        batch = {"input_ids": masked_inputs,
                 "labels": original_input,
                 "token_type_ids": token_type_ids,
                 "next_sentence_label": next_sentence_label,
                 "attention_mask": attention_mask}
        return batch

    def torch_mask_tokens(self, inputs: LongTensor) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """

        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = full(labels.shape, self.mlm_probability)
        special_tokens_mask = torch.isin(inputs, self.special_tokens)

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.mask_token

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(self.vocab_size, labels.shape).long()
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


class DataCollatorClaFinetune:
    def __init__(self, pad_token: int):
        """Collator for classification.
        Input_ids will be padded with the pad token given.

        :param pad_token: pas token
        """
        self.pad_token = pad_token

    def __call__(self, examples: List[Dict[str, Any]], return_tensors=None) -> Dict[str, LongTensor]:
        x, y = _collate_cla(examples, self.pad_token)
        attention_mask = (x != self.pad_token).int()
        return {"input_ids": x, "labels": y, "attention_mask": attention_mask}


class DataCollatorContrastive:
    def __init__(self, pad_token: int, bos_token: int, eos_token: int):
        """Collator for contrastive learning.
        It first pads the batch, then repeat it one time along the batch dimension.
        The labels are ranks (arange()).

        :param pad_token: pas token
        """
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token

    def __call__(self, batch: List[Dict[str, Any]], return_tensors=None) -> Dict[str, LongTensor]:
        _add_bos_eos_tokens_to_batch(batch, bos_tok=self.bos_token, eos_tok=self.eos_token)
        x = _pad_batch(batch, self.pad_token)  # .repeat(2, 1)  # (N*2,T)
        attention_mask = (x != self.pad_token).int()
        return {"input_ids": x, "labels": arange(x.size(0)).long(), "attention_mask": attention_mask}  # rank


class DataCollatorContrastiveSupervised:
    def __init__(self, pad_token: int, bos_token: int, eos_token: int, tokenizer: MIDITokenizer, base_offsets):
        """Collator for contrastive learning.
        It first pads the batch, then repeat it one time along the batch dimension.
        The labels are ranks (arange()).

        :param pad_token: pas token
        """
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.tokenizer = tokenizer
        self.base_offsets = base_offsets

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, LongTensor]:
        _add_bos_eos_tokens_to_batch(batch, bos_tok=self.bos_token, eos_tok=self.eos_token)
        x = _pad_batch(batch, self.pad_token)

        valid_offsets = self.get_possible_offsets(x, self.base_offsets)
        offsets = [[] for _ in range(len(x))]
        for name_, valid_samples in valid_offsets.items():
            for valid_sample in valid_samples:
                offsets[valid_sample].append(self.base_offsets[name_])

        x2 = []
        current_seq_idx = 0
        for possible_offsets in offsets:
            if len(possible_offsets) == 0:  # this sequence cannot be augmented, remove it from batch
                x = cat([x[:current_seq_idx], x[current_seq_idx+1:]])
                continue
            x2.append(LongTensor(self.data_augmentation_tokens(x[current_seq_idx].numpy(), choice(possible_offsets))))
            current_seq_idx += 1

        x2 = torch.nn.utils.rnn.pad_sequence(x2, batch_first=True)
        attention_mask = (x != self.pad_token).int()
        attention_mask2 = (x2 != self.pad_token).int()
        return {"input_ids": x,
                "input_ids2": x2,
                "labels": arange(x.size(0)).long(),
                "attention_mask": attention_mask,
                "attention_mask2": attention_mask2}

    def get_possible_offsets(self, samples: LongTensor, offsets: Dict) -> Dict:
        possible_offsets = {name: [] for name in offsets}

        # Decode BPE if needed
        if self.tokenizer.has_bpe:
            samples_no_bpe = []
            for sample in samples:
                seq = TokSequence(ids=sample.tolist(), ids_bpe_encoded=True)
                self.tokenizer.decode_bpe(seq)
                samples_no_bpe.append(LongTensor(seq.ids))
            # Need to pad as lengths have changed
            samples = torch.nn.utils.rnn.pad_sequence(samples_no_bpe, True, self.pad_token)

        # Get min and max pitches
        pitch_ids_vocab = cat([LongTensor(self.tokenizer.token_ids_of_type("Pitch")),
                               LongTensor(self.tokenizer.token_ids_of_type("NoteOn"))])
        for si, sample in enumerate(samples):
            ids_pitch = sample[torch.isin(sample, pitch_ids_vocab)]
            min_id, max_id = min(ids_pitch), max(ids_pitch)
            min_pitch = int(self.tokenizer[int(min_id)].split("_")[1])
            max_pitch = int(self.tokenizer[int(max_id)].split("_")[1])

            # Possible offsets
            for name, offset in offsets.items():
                if self.tokenizer.pitch_range.start <= min_pitch + offset[0] < self.tokenizer.pitch_range.stop and \
                        self.tokenizer.pitch_range.start <= max_pitch + offset[0] < self.tokenizer.pitch_range.stop:
                    possible_offsets[name].append(si)

        return possible_offsets

    def data_augmentation_tokens(
            self,
            tokens: Union[LongTensor, List[int]],
            offsets: Tuple[int, int, int],
    ) -> List[int]:
        pitch_offset, vel_offset = offsets[:2]

        # Decode BPE
        bpe_decoded = False
        if self.tokenizer.has_bpe:
            in_seq = TokSequence(ids=tokens.tolist() if isinstance(tokens, LongTensor) else tokens,
                                 ids_bpe_encoded=True)
            self.tokenizer.decode_bpe(in_seq)
            tokens = in_seq.ids
            bpe_decoded = True

        # Converts to np array if necessary
        if not isinstance(tokens, LongTensor):
            tokens = LongTensor(tokens)
        augmented = tokens.clone()

        if pitch_offset != 0:
            # Get the maximum and lowest pitch in original track
            pitch_tokens = cat([LongTensor(self.tokenizer.token_ids_of_type("Pitch")),
                                LongTensor(self.tokenizer.token_ids_of_type("NoteOn"))])
            mask_pitch = torch.isin(tokens, pitch_tokens)

            # Perform augmentation on pitch
            augmented[mask_pitch] += pitch_offset

        # Velocity augmentation
        if vel_offset != 0:
            vel_tokens = LongTensor(self.tokenizer.token_ids_of_type("Velocity"))

            mask = torch.isin(augmented, vel_tokens)

            augmented[mask] += vel_offset
            augmented[mask] = torch.clip(augmented[mask], vel_tokens[0], vel_tokens[-1])

        # Convert array to list and reapply BPE if necessary
        seq = TokSequence(ids=augmented.tolist())
        if bpe_decoded:
            self.tokenizer.apply_bpe(seq)

        return seq.ids


class DataCollatorTranscription:
    def __init__(self, pad_token: int, bos_token: int, eos_token: int):
        """Collator for music transcription.
        Input_ids will be padded with the pad token given.

        :param pad_token: pad token
        :param bos_token: bos token
        :param eos_token: eos token
        """
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token

    def __call__(self, examples: List[Dict[str, Any]], return_tensors=None) -> Dict[str, LongTensor]:
        _add_bos_eos_tokens_to_batch(examples, "decoder_input_ids", self.bos_token, self.eos_token)

        input_embed = stack([e["inputs_embeds"] for e in examples])
        labels = _pad_batch(examples, -100, "decoder_input_ids", examples[0]["pad_on_left"])
        model_kwargs = {"inputs_embeds": input_embed}

        if examples[0]["decoder_inputs"]:
            decoder_input_ids = _pad_batch(examples, 0, "decoder_input_ids", examples[0]["pad_on_left"])[:, :-1]
            labels = labels[:, 1:]
            decoder_attention_mask = (decoder_input_ids != self.pad_token).int()
            model_kwargs["decoder_input_ids"] = decoder_input_ids.contiguous()
            model_kwargs["decoder_attention_mask"] = decoder_attention_mask.contiguous()
        model_kwargs["labels"] = labels.contiguous()

        return model_kwargs


def _add_bos_eos_tokens_to_batch(
        batch: List[Dict[str, LongTensor]],
        dict_key: str = "input_ids",
        bos_tok: int = None,
        eos_tok: int = None
):
    if bos_tok is None and eos_tok is None:
        return

    for i in range(len(batch)):
        if bos_tok is not None and eos_tok is not None:
            batch[i][dict_key] = cat([LongTensor([bos_tok]), batch[i][dict_key], LongTensor([eos_tok])]).long()
        elif bos_tok is not None:
            batch[i][dict_key] = cat([LongTensor([bos_tok]), batch[i][dict_key]]).long()
        else:  # EOS not None
            batch[i][dict_key] = cat([batch[i][dict_key], LongTensor([eos_tok])]).long()


def _pad_batch(
        batch: List[Dict[str, LongTensor]],
        pad_token: int,
        dict_key: str = "input_ids",
        pad_on_left: bool = False
) -> LongTensor:
    """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""

    length_of_first = batch[0][dict_key].size(0)

    # Check if padding is necessary.
    are_tensors_same_length = all(x[dict_key].size(0) == length_of_first for x in batch)
    if are_tensors_same_length:
        return stack([e[dict_key] for e in batch], dim=0).long()

    # Creating the full tensor and filling it with our data.
    if pad_on_left:
        return pad_left([e[dict_key] for e in batch], pad_token)
    else:
        return torch.nn.utils.rnn.pad_sequence(
            [e[dict_key] for e in batch],
            batch_first=True,
            padding_value=pad_token
        ).long()


def pad_left(batch: List[LongTensor], pad_token: int) -> LongTensor:
    # Here the sequences are padded to the left, so that the last token along the time dimension
    # is always the last token of each seq, allowing to efficiently generate by batch
    batch = [torch.flip(seq, dims=(0,)) for seq in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=pad_token)  # (N,T)
    batch = torch.flip(batch, dims=(1,)).long()
    return batch


def _nsp_swap(
        examples: List[Dict[str, LongTensor]],
        nsp_ratio: float,
        seq_b_ratio: float,
        sep_token: Union[int, LongTensor],
        pad_token: Union[int, LongTensor],
) -> Tuple[List[Dict[str, LongTensor]], LongTensor, LongTensor]:
    # Determine idx to mix for NSP
    nb_mixed = int(len(examples) * nsp_ratio)
    token_type_ids = [full((len(examples[idx]["input_ids"]) + 1,), 0).long() for idx in range(len(examples))]
    next_sentence_label = full((len(examples),), 0).long()
    new_next_idx = arange(len(examples))
    if nb_mixed > 1:
        # create permutations / pairs of swapped seq_a - seq_b
        permutations = torch.randperm(nb_mixed)
        while any(equal := permutations == arange(nb_mixed)):
            idx_equal = torch.where(equal)[0]  # list of idx to mix together
            if len(idx_equal) == 1:
                idx_to_swap = torch.multinomial(cat([permutations[:idx_equal[0]],
                                                     permutations[idx_equal[0] + 1:]]).float(), 1).long()
                permutations[idx_equal[0]] = idx_to_swap
                permutations[idx_to_swap] = idx_equal[0]
            else:
                permutations[idx_equal] = permutations[idx_equal[torch.randperm(len(idx_equal))]]  # only betw those eq
        samples_to_mix_idx = torch.multinomial(arange(len(examples)).float(), nb_mixed).long()
        new_next_idx[samples_to_mix_idx] = samples_to_mix_idx.clone()[permutations]

    # Swap them and prepare labels / token_type_ids
    # SEP token need to be added at the end (before padding) as we may otherwise swap sequences of
    # different lengths and add additional SEP tokens to some
    examples_copy = [e["input_ids"].clone() for e in examples]
    for idx, idx_next in enumerate(new_next_idx):
        sep_idx = int(len(examples[idx]["input_ids"]) * seq_b_ratio)
        len_seq_b = len(examples[idx]["input_ids"]) - sep_idx
        len_next_seq = len(examples[idx_next]["input_ids"]) - 1  # -1 because of BOS token
        if len_seq_b > len_next_seq:
            sep_idx = len(examples[idx]["input_ids"]) - len_next_seq
            len_seq_b = len_next_seq
        token_type_ids[idx] = cat([token_type_ids[idx][:sep_idx + 1], full((len_seq_b,), 1).long()]).long()
        if idx != idx_next:  # meaning seq_b is not seq_a's second part
            next_sentence_label[idx] = 1

        examples[idx]["input_ids"] = cat([examples_copy[idx][:sep_idx],
                                          sep_token,
                                          examples_copy[idx_next][-len_seq_b:]]).long()
        examples[idx]["labels"] = cat([examples_copy[idx][:sep_idx],
                                       sep_token,
                                       examples_copy[idx_next][-len_seq_b:]]).long()

    token_type_ids = torch.nn.utils.rnn.pad_sequence(token_type_ids, True, float(pad_token)).long()  # stack and pad
    return examples, token_type_ids, next_sentence_label


def _collate_cla(batch: List[Dict[str, Union[LongTensor, int]]], pad_tok: int) -> Tuple[LongTensor, LongTensor]:
    x = _pad_batch(batch, pad_tok)
    y = LongTensor([d["labels"] for d in batch])
    return x, y  # (N,T) and (N)
