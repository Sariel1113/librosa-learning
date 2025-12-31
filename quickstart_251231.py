import numpy as np
import librosa

# 1. Get the file path to an included audio example
filename = librosa.example('nutcracker')

y, sr = librosa.load(filename)

hop_length = 512

y_harmonic, y_percussive = librosa.effects.hpss(y)

tempo, beat_frames = librosa.beat.beat_track(y=y_percussive, sr=sr)
tempo = float(tempo.item())

mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13)
mfcc_delta = librosa.feature.delta(mfcc)
beat_mfcc_delta = librosa.util.sync(np.vstack([mfcc, mfcc_delta]), beat_frames)

chromagram = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)
beat_chroma = librosa.util.sync(chromagram, beat_frames, aggregate=np.median)
beat_features = np.vstack([beat_chroma, beat_mfcc_delta])

print