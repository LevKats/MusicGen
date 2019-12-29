from feature_extractor import FeatureExtractor, Type
from chain_model import ChainModel
import numpy as np

from mido import Message
from mido import MidiFile
from mido import MidiTrack


def main():
    extractor = FeatureExtractor("Соната 4.mid")
    extractor.parse()
    extractor.power2_decomposition()
    print(len(extractor.features))
    print("\n".join(map(str, extractor.features)))
    features = extractor.encode_features(2)
    model = ChainModel()
    model.fit(features)
    res = model.predict(200, np.random.choice(
        model.proba_matrix.nonzero()[0], 1)[0]
                        )
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    time = 0.0
    for num in res:
        decoded_feature = extractor.coder.decode(num)[0]
        if decoded_feature.type == Type.NOTE:
            track.append(Message(
                'note_on', note=decoded_feature.note, velocity=64, time=int(time)
            ))
            track.append(Message(
                'note_off', note=decoded_feature.note, velocity=127,
                time=int(decoded_feature.duration)
            ))
            time = 0.0
        else:
            time = decoded_feature.duration

    mid.save('new_song.mid')


if __name__ == "__main__":
    main()
