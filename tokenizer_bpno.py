""" Similar to REMI but with Note-Off tokens instead of Duration

"""

from typing import List, Tuple, Dict, Optional, Union

from miditoolkit import Instrument, Note, TempoChange
from miditok import MIDITokenizer, Event, TokSequence
from miditok.midi_tokenizer import _in_as_seq, _out_as_complete_seq
from miditok.constants import MIDI_INSTRUMENTS
from miditok.utils import detect_chords

from constants import PITCH_RANGE, NB_VELOCITIES, ADDITIONAL_TOKENS, BEAT_RES, TIME_DIVISION, SPECIAL_TOKENS


class BarPosNoteOff(MIDITokenizer):

    def __init__(self, pitch_range: range = PITCH_RANGE, beat_res: Dict[Tuple[int, int], int] = BEAT_RES,
                 nb_velocities: int = NB_VELOCITIES, additional_tokens: Dict[str, Union[bool, int]] = ADDITIONAL_TOKENS,
                 special_tokens=SPECIAL_TOKENS, params=None):
        super().__init__(pitch_range, beat_res, nb_velocities, additional_tokens, special_tokens, False, params)

    @_out_as_complete_seq
    def track_to_tokens(self, track: Instrument) -> TokSequence:
        """ Converts a track (miditoolkit.Instrument object) into a sequence of tokens
        :param track: MIDI track to convert
        :return: sequence of corresponding tokens
        """
        # Make sure the notes are sorted first by their onset (start) times, second by pitch
        # notes.sort(key=lambda x: (x.start, x.pitch))  # done in midi_to_tokens
        ticks_per_sample = self._current_midi_metadata['time_division'] / max(self.beat_res.values())
        ticks_per_bar = self._current_midi_metadata['time_division'] * 4
        events = []

        # Creates the Note On, Note Off and Velocity events
        for n, note in enumerate(track.notes):
            # Note-On / Velocity / Note-Off
            events.append(Event(type='NoteOn', time=note.start, value=note.pitch, desc=note.end))
            events.append(Event(type='Velocity', time=note.start, value=note.velocity, desc=f'{note.velocity}'))
            events.append(Event(type='NoteOff', time=note.end, value=note.pitch, desc=note.end))

        # Sorts events
        events.sort(key=lambda x: x.time)

        # Bar / Pos
        previous_tick = 0
        previous_note_end = track.notes[0].start + 1
        current_bar = -1
        for e, event in enumerate(events.copy()):
            if event.time == previous_tick:
                if event.type == 'NoteOn':
                    previous_note_end = max(previous_note_end, event.desc)
                continue

            # Bar
            nb_new_bars = event.time // ticks_per_bar - current_bar
            for i in range(nb_new_bars):
                events.append(Event(type='Bar', time=(current_bar + i + 1) * ticks_per_bar, value='None', desc=0))
            current_bar += nb_new_bars

            # Position
            pos_index = int((event.time % ticks_per_bar) / ticks_per_sample)
            events.append(Event(type='Position', time=event.time, value=pos_index, desc=event.time))

            if event.type == 'NoteOn':
                previous_note_end = max(previous_note_end, event.desc)
            previous_tick = event.time

        # Adds chord events if specified
        if self.additional_tokens['Chord'] and not track.is_drum:
            events += detect_chords(track.notes, self._current_midi_metadata['time_division'], self._first_beat_res)

        events.sort(key=lambda x: (x.time, self._order(x)))

        return TokSequence(events=events)

    @_in_as_seq()
    def tokens_to_track(self, tokens: Union[TokSequence, List], time_division: Optional[int] = TIME_DIVISION,
                        program: Optional[Tuple[int, bool]] = (0, False), default_duration: int = None) \
            -> Tuple[Instrument, List[TempoChange]]:
        ticks_per_pos = time_division // max(self.beat_res.values())
        ticks_per_bar = time_division * 4
        events = []
        for token in tokens.tokens:
            tok_type, tok_val = token.split('_')
            events.append(Event(type=tok_type, value=tok_val))

        max_duration = self.durations[-1][0] * time_division + self.durations[-1][1] * (time_division //
                                                                                        self.durations[-1][2])
        name = 'Drums' if program[1] else MIDI_INSTRUMENTS[program[0]]['name']
        instrument = Instrument(program[0], is_drum=program[1], name=name)
        tempo_changes = [TempoChange(120, -1)]  # mock the first tempo change to optimize below

        current_tick = 0
        ei = 0
        current_bar = -1
        while ei < len(events):
            if events[ei].type == 'NoteOn':
                try:
                    if events[ei + 1].type == 'Velocity':
                        pitch = int(events[ei].value)
                        vel = int(events[ei + 1].value)

                        # look for an associated note off event to get duration
                        offset_tick = 0
                        offset_bar = 0
                        duration = 0
                        for i in range(ei + 1, len(events)):
                            if events[i].type == 'NoteOff' and int(events[i].value) == pitch:
                                duration = offset_tick
                                break
                            elif events[i].type == 'Bar':
                                offset_bar += 1
                            elif events[i].type == 'Position':
                                offset_tick = (0 if current_bar < 0 else current_bar + offset_bar) * ticks_per_bar + \
                                              int(events[i].value) * ticks_per_pos - current_tick
                            if offset_tick > max_duration:  # will not look for Note Off beyond
                                break

                        if duration == 0 and default_duration is not None:
                            duration = default_duration
                        if duration > 0:
                            instrument.notes.append(Note(vel, pitch, current_tick, current_tick + duration))
                        ei += 1
                except IndexError as _:
                    pass
            if events[ei].type == 'Bar':
                current_bar += 1
                current_tick = current_bar * ticks_per_bar
            elif events[ei].type == 'Position':
                if current_bar == -1:
                    current_bar = 0  # as this Position token occurs before any Bar token
                current_tick = current_bar * ticks_per_bar + int(events[ei].value) * ticks_per_pos
            ei += 1
        if len(tempo_changes) > 1:
            del tempo_changes[0]  # delete mocked tempo change
        tempo_changes[0].time = 0
        return instrument, tempo_changes

    def _create_base_vocabulary(self, *args, **kwargs) -> List[str]:
        """ Creates the Vocabulary object of the tokenizer.
        See the docstring of the Vocabulary class for more details about how to use it.
        NOTE: token index 0 is often used as a padding index during training
        :return: the vocabulary object
        """
        vocab = ['Bar_None']

        # NOTE ON
        vocab += [f'NoteOn_{i}' for i in self.pitch_range]

        # NOTE OFF
        vocab += [f'NoteOff_{i}' for i in self.pitch_range]

        # VELOCITY
        vocab += [f'Velocity_{i}' for i in self.velocities]

        # POSITION
        nb_positions = max(self.beat_res.values()) * 4  # 4/4 time signature
        vocab += [f'Position_{i}' for i in range(nb_positions)]

        return vocab

    def _create_token_types_graph(self) -> Dict[str, List[str]]:
        """ Returns a graph (as a dictionary) of the possible token
        types successions.
        NOTE: Program type is not referenced here, you can add it manually by
        modifying the tokens_types_graph class attribute following your strategy.
        :return: the token types transitions dictionary
        """
        dic = dict()

        dic['Bar'] = ['Position', 'Bar']

        dic['Position'] = ['NoteOn', 'NoteOff']
        dic['NoteOn'] = ['Velocity']
        dic['Velocity'] = ['NoteOn', 'Position', 'Bar']
        dic['NoteOff'] = ['NoteOff', 'NoteOn', 'Position', 'Bar']

        return dic

    @staticmethod
    def _order(x: Event) -> int:
        """ Helper function to sort events in the right order
        :param x: event to get order index
        :return: an order int
        """
        if x.type == 'Program':
            return 0
        elif x.type == 'Bar':
            return 1
        elif x.type == 'Position':
            return 2
        elif x.type == 'NoteOff':
            return 3
        else:  # for other types of events, the order should be handle when inserting the events in the sequence
            return 4
