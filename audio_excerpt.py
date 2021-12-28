import librosa
import pydub
import os
import pandas as pd
import numpy as np
from containers import *

fileDir = os.path.dirname(os.path.realpath('__file__'))
print(fileDir)

def extract_audio(csv_path):
    total_df = pd.read_csv(csv_path)

    for index, row in total_df.iterrows():
        count = 0
        start_time = row['start_time']
        end_time = row['end_time']
        file_name = row['filename']
        json_name = file_name.split('/')[-1]
        sentence = row['sentence']
        y, sr = librosa.load(file_name, sr=16000, duration=10)
        audio_seg = y[start_time:end_time]
        audio_excerpt_name = json_name + count
        write(audio_excerpt_name, sr, audio_seg, normalized=False)
        count += 1

def write(audio_excerpt_name, sr, x, normalized=False):
    """numpy array to MP3"""
    channels = 2 if (x.ndim == 2 and x.shape[1] == 2) else 1
    if normalized:  # normalized array - each item should be a float in [-1, 1)
        y = np.int16(x * 2 ** 15)
    else:
        y = np.int16(x)
    song = pydub.AudioSegment(y.tobytes(), frame_rate=sr, sample_width=2, channels=channels)
    audio_excerpt_name_path = os.path.join(fileDir, 'audio_excerpts', audio_excerpt_name) + '.mp3'
    song.export(audio_excerpt_name_path, format="mp3", bitrate="320k")

# def extract_segments(y, sr, segments):
#     # compute segment regions in number of samples
#     starts = np.floor(segments.start_time.dt.total_seconds() * sr).astype(int)
#     ends = np.ceil(segments.end_time.dt.total_seconds() * sr).astype(int)
#
#     # slice the audio into segments
#     for start, end in zip(starts, ends):
#         audio_seg = y[start:end]
#         print('extracting audio segment:', len(audio_seg), 'samples')
#         print(type(audio_seg))


