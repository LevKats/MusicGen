import mido
from collections import namedtuple
from enum import Enum


Feature = namedtuple('Feature', ('type', 'duration', 'note'))


class Type(Enum):
    NOTE = 1
    PAUSE = 2


def get_ngrams(seq, n):
    array = list(seq)
    return zip(*[array[i:] for i in range(n)])


class FeaturesToInt:
    def __init__(self):
        self.__dct = {}
        self.__inv_dct = {}
        self.__index = 0

    def encode(self, tup):
        if tup in self.__inv_dct:
            return self.__inv_dct[tup]
        self.__dct[self.__index] = tup
        self.__inv_dct[tup] = self.__index
        self.__index += 1
        return self.__index - 1

    def decode(self, num):
        return self.__dct[num]


class FeatureExtractor:
    def __init__(self, filename):
        self.mid = mido.MidiFile(filename)
        self.features = None
        self.coder = None
        self.encoded_features = None

    def parse(self):
        all_messages = []
        time = 0.0
        for i, track in enumerate(self.mid.tracks):
            for msg in track:
                time += msg.time
                if msg.type in ["note_on", "note_off"]:
                    msg.time = time
                    all_messages.append(msg)

        current_notes = {}
        time = 0.0
        self.features = []
        for msg in all_messages:
            if msg.type == "note_on":
                if msg.time != time and len(current_notes) == 0:
                    self.features.append(Feature(
                        type=Type.PAUSE, duration=msg.time - time, note=None
                    ))
                if msg.note not in current_notes:
                    current_notes[msg.note] = msg
                time = msg.time

            elif msg.type == "note_off" and msg.note in current_notes:
                time = msg.time
                max_note = max(current_notes)
                if msg.note == max_note:
                    self.features.append(Feature(
                        type=Type.NOTE, note=msg.note,
                        duration=msg.time - current_notes[max_note].time,
                    ))
                current_notes.pop(msg.note)
            else:
                pass

    def encode_features(self, order=2):
        self.coder = FeaturesToInt()
        self.encoded_features = tuple(get_ngrams(map(
            lambda tup: self.coder.encode(tup),
            get_ngrams(self.features, order)
        ), 2))
        return self.encoded_features
