import librosa
import pydub
import os
import pandas as pd
import numpy as np
from containers import *

fileDir = os.path.dirname(os.path.realpath(__file__))


# def extract_audio(csv_path, audio_path):
#     total_df = pd.read_csv(csv_path)
#
#     for index, row in total_df.iterrows():
#         start_time = str(row['start_time'][:-1])
#         end_time = str(row['end_time'][:-1])
#         file_name = row['filename']
#         json_name = file_name.split('\\')[-1]
#         audio_name = json_name.rsplit('.', 1)[0] + '.ogg'
#         sentence = row['sentence']
#         audio_file_name = os.path.join(audio_path, audio_name)
#         ogg_files = [f for f in os.listdir(audio_path) if
#                       os.path.isfile(os.path.join(audio_path, f)) and f.endswith(".ogg")]
#         if audio_name in ogg_files:
#             print('found one match!')
#             y, sr = librosa.load(audio_file_name, sr=16000, duration=10)
#             audio_seg = y[start_time:end_time]
#             audio_excerpt_name = json_name + start_time
#             write(audio_excerpt_name, sr, audio_seg, normalized=False)



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

csv_path = 'confidence_css.csv'
audio_path = os.path.join(fileDir, 'audio')

#extract_audio(csv_path, audio_path)


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


def read_data(csv_path):
    df = pd.read_csv(csv_path, sep=',')

    # Use proper pd datatypes
    df['start_time'] = pd.to_datetime(df['start_time'], unit='s')
    df['start_time'] = df['start_time'].str.replace('s', '')
    df['end_time'][:-1]= pd.to_timedelta(df['end_time'], unit='s')
    df['end_time'] = df['end_time'].str.replace('s', '')
    df['json_name'] = df['filename'].split('\\')[-1]
    df['audio_name'] = df['json_name'].rsplit('.', 1)[0] + '.ogg'


    return df

def extract_segments(sr, df):
    # compute segment regions in number of samples
    starts = np.floor(df.start_time.dt.total_seconds() * sr).astype(int)
    ends = np.ceil(df.end_time.dt.total_seconds() * sr).astype(int)
    audio_names = df.audio_name.astype(str)

    # slice the audio into segments
    for start, end, audio_name in zip(starts, ends, audio_names):
        audio_file_path = os.path.join(audio_path, audio_name)
        y, sr = librosa.load(audio_file_path, sr=sr, duration=10)
        audio_seg = y[start:end]
        audio_excerpt_name = str(audio_name + start)
        write(audio_excerpt_name, sr, audio_seg, normalized=False)

def extract_all_audios(csv_path, audio_path, sr):
    df = read_data(csv_path)
    ogg_files = [f for f in os.listdir(audio_path) if
                 os.path.isfile(os.path.join(audio_path, f)) and f.endswith(".ogg")]
    extract_segments(sr, df)