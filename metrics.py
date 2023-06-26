
from typing import Tuple, List

from miditok import MIDITokenizer, TokSequence


def tse(tokens: List[int], tokenizer: MIDITokenizer) -> Tuple[float, float, float, float, float]:
    r"""Checks if a sequence of tokens is made of good token types
    successions and returns the error ratio (lower is better).
    The common implementation in MIDITokenizer class will check token types,
    duplicated notes and time errors. It works for REMI, TSD and Structured.
    Other tokenizations override this method to include other errors
    (like no NoteOff / NoteOn for MIDILike and embedding pooling).
    Overridden methods must call decompose_bpe at the beginning if BPE is used!

    :param tokens: sequence of tokens to check.
    :param tokenizer
    :return: the error ratio (lower is better).
    """
    nb_tok_predicted = len(tokens)  # used to norm the score
    tokens = TokSequence(ids=tokens, ids_bpe_encoded=tokenizer.has_bpe)
    if tokenizer.has_bpe:
        tokenizer.decode_bpe(tokens)
    tokenizer.complete_sequence(tokens)
    tokens = tokens.tokens

    err_type = 0  # i.e. incompatible next type predicted
    err_time = 0  # i.e. goes back or stay in time (does not go forward)
    err_ndup = 0
    err_nnon = 0  # note-off predicted while not being played
    err_nnof = 0  # note-on predicted with no note-off to end it
    previous_type = tokens[0].split("_")[0]
    current_pos = -1
    notes_being_played = {pitch: 0 for pitch in range(0, 128)}
    pitches_current_moment = []  # only at the current position / time step - used for ndup
    note_tokens_types = ["Pitch", "NoteOn"]
    pos_per_beat = max(tokenizer.beat_res.values())
    max_duration = tokenizer.durations[-1][0] * pos_per_beat
    max_duration += tokenizer.durations[-1][1] * (pos_per_beat // tokenizer.durations[-1][2])

    # Init first note and current pitches if needed
    if previous_type in note_tokens_types:
        notes_being_played[int(tokens[0].split("_")[1])] += 1
        pitches_current_moment.append(int(tokens[0].split("_")[1]))
    elif previous_type == "Position":
        current_pos = int(tokens[0].split("_")[1])
    del tokens[0]

    for i, token in enumerate(tokens):
        event_type, event_value = token.split("_")

        # Good token type
        if event_type in tokenizer.tokens_types_graph[previous_type]:
            if event_type == "Bar":  # reset
                current_pos = -1
                pitches_current_moment = []

            elif event_type == "Position":
                if int(event_value) <= current_pos and previous_type != "Rest":
                    err_time += 1  # token position value <= to the current position
                current_pos = int(event_value)
                pitches_current_moment = []

            elif event_type == "TimeShift":
                pitches_current_moment = []

            elif event_type in note_tokens_types:  # checks if not already played and/or that a NoteOff is associated
                pitch_val = int(event_value)
                if pitch_val in pitches_current_moment:
                    err_ndup += 1  # pitch already played at current position
                pitches_current_moment.append(pitch_val)
                if event_type == "NoteOn":
                    # look for an associated note off token to get duration
                    offset_sample = 0
                    offset_bar = 0
                    for j in range(i + 1, len(tokens)):
                        event_j_type, event_j_value = tokens[j].split("_")[0], tokens[j].split("_")[1]
                        if event_j_type == 'NoteOff' and int(event_j_value) == pitch_val:
                            notes_being_played[pitch_val] += 1
                            break  # all good
                        elif event_j_type == 'Bar':
                            offset_bar += 1
                        elif event_j_type == 'Position':
                            if offset_bar == 0:
                                offset_sample = int(event_j_value) - current_pos
                            else:
                                offset_sample = pos_per_beat - current_pos + (offset_bar - 1) * pos_per_beat * 4 + \
                                                int(event_j_value)
                        elif event_j_type == 'TimeShift':
                            offset_sample += tokenizer._token_duration_to_ticks(event_j_value, pos_per_beat)
                        if offset_sample > max_duration:  # will not look for Note Off beyond
                            err_nnof += 1
                            break

            elif event_type == "NoteOff":
                if notes_being_played[int(event_value)] == 0:
                    err_nnon += 1  # this note wasn't being played
                else:
                    notes_being_played[int(event_value)] -= 1
        # Bad token type
        else:
            err_type += 1
        previous_type = event_type

    return tuple(map(lambda err: err / nb_tok_predicted, (err_type, err_time, err_ndup, err_nnon, err_nnof)))
