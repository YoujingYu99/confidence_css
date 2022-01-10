import librosa
import pydub
import os
import pandas as pd
import numpy as np
from pydub import AudioSegment
from containers import *

AudioSegment.converter = r"C:\ffmpeg\bin\ffmpeg.exe"
AudioSegment.ffprobe = r"C:\ffmpeg\bin\ffprobe.exe"
AudioSegment.ffmpeg = r"C:\ffmpeg\bin\ffmpeg.exe"
pydub.utils.get_prober_name = lambda: r"C:\ffmpeg\bin\ffprobe.exe"
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



# def write(audio_excerpt_name, sr, x, normalized=False):
#     """numpy array to MP3"""
#     channels = 2 if (x.ndim == 2 and x.shape[1] == 2) else 1
#     if normalized:  # normalized array - each item should be a float in [-1, 1)
#         y = np.int16(x * 2 ** 15)
#     else:
#         y = np.int16(x)
#     song = pydub.AudioSegment(y.tobytes(), frame_rate=sr, sample_width=2, channels=channels)
#     audio_excerpt_name_path = os.path.join(fileDir, 'audio_excerpts', audio_excerpt_name) + '.mp3'
#     song.export(audio_excerpt_name_path, format="mp3", bitrate="320k")

csv_path = 'confidence_css.csv'
audio_path = os.path.join(fileDir, 'audio')
excerpt_output_path = os.path.join(fileDir, 'audio_excerpts')



def read_data(csv_path):
    df = pd.read_csv(csv_path, sep=',')

    # Use proper pd datatypes
    df['start_time'] = df['start_time'].str.replace('s', '')
    df['start_time'] = df['start_time'].astype(float)
    df['end_time'] = df['end_time'].str.replace('s', '')
    # leave one second buffer
    df['end_time'] = df['end_time'].astype(float) + 1.0
    filename_list = df['filename'].tolist()
    json_name_list = [jname.split('\\')[-1] for jname in filename_list]
    audio_name_list = [aname.rsplit('.', 1)[0] + '.ogg' for aname in json_name_list]
    df['audio_name'] = audio_name_list

    return df

def extract_segments(df, ogg_files, excerpt_output_path):
    starts = df.start_time.astype(float)
    ends = df.end_time.astype(float)
    audio_names = df.audio_name.astype(str)

    # slice the audio into segments
    for start, end, audio_name in zip(starts, ends, audio_names):
        if audio_name in ogg_files:
            #print(audio_name)
            audio_file_path = os.path.join(audio_path, audio_name)
            audio_excerpt_name = os.path.join(excerpt_output_path, str(audio_name + str(start)))
            print(audio_excerpt_name)
            # working in milliseconds
            start = start * 1000
            end = end * 1000
            newAudio = AudioSegment.from_ogg(audio_file_path)
            newAudio = newAudio[start:end]
            newAudio.export(audio_excerpt_name + '.ogg', format="ogg")

def extract_all_audios(csv_path, audio_path, excerpt_output_path,):
    df = read_data(csv_path)
    ogg_files = [f for f in os.listdir(audio_path) if
                 os.path.isfile(os.path.join(audio_path, f)) and f.endswith(".ogg")]
    extract_segments(df, ogg_files, excerpt_output_path)


extract_all_audios(csv_path, audio_path, excerpt_output_path,)