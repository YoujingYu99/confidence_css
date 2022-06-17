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


csv_path = 'confidence_css2.csv'
audio_path = os.path.join(fileDir, 'audio')
excerpt_output_path = os.path.join(fileDir, 'audio_excerpts')



def read_data(csv_path):
    df = pd.read_csv(csv_path, sep=',')

    # Use proper pd datatypes
    df['start_time'] = df['start_time'].str.replace('s', '')
    df['start_time'] = df['start_time'].astype(float)
    df['sent_end_time'] = df['sent_end_time'].str.replace('s', '')
    df['sent_end_time'] = df['sent_end_time'].astype(float)
    filename_list = df['filename'].tolist()
    json_name_list = [jname.split('\\')[-1] for jname in filename_list]
    audio_name_list = [aname.rsplit('.', 1)[0] + '.ogg' for aname in json_name_list]
    df['audio_name'] = audio_name_list

    return df

def extract_segments(df, ogg_files, excerpt_output_path):
    starts = df.start_time.astype(float)
    ends = df.sent_end_time.astype(float)
    audio_names = df.audio_name.astype(str)

    # slice the audio into segments
    for start, end, audio_name in zip(starts, ends, audio_names):
        if audio_name in ogg_files:
            # print(audio_name)
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