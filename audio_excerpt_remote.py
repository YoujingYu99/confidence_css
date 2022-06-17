"""Extract audio segments from their parent audio files.

This script uses information on the questions from the csv file to extract
short audio segments from the parent audio files.
"""

import pydub
import os
import pandas as pd
from pydub import AudioSegment


# AudioSegment.converter = r'/ffmpeg/bin/ffmpeg.exe'
# AudioSegment.ffprobe = r'/ffmpeg/bin/ffprobe.exe'
# AudioSegment.ffmpeg = r'/ffmpeg/bin/ffmpeg.exe'
# pydub.utils.get_prober_name = lambda: r'ffmpeg/bin/ffprobe.exe'


# home_dir is the location of script
home_dir = os.path.join('/home', 'yyu')

csv_path = os.path.join(home_dir, 'data_sheets', 'confidence_dataframe_0.csv')
# Audio data file
fileDir = os.path.join(home_dir, 'data')
audio_path = os.path.join(
    fileDir, 'Spotify-Podcasts', 'podcasts-audio-only-2TB', 'podcasts-audio', '0'
)
excerpt_output_path = os.path.join(home_dir, 'extracted_audio', '0')


# AudioSegment.converter = os.path.join(home_dir, 'confidence_css', 'ffmpeg.exe')
#
# AudioSegment.ffprobe = os.path.join(home_dir, 'confidence_css', 'ffprobe.exe')
# AudioSegment.ffmpeg = os.path.join(home_dir, 'confidence_css', 'ffmpeg.exe')
# pydub.utils.get_prober_name = lambda: os.path.join(
#     home_dir, 'confidence_css', 'ffprobe.exe'
# )


def read_data(csv_path):
    """
    :param csv_path: Path of the csv file containing information about
    the questions.
    :return: A dataframe containing 'start_time', 'sent_end_time' and
    'filename'.
    """
    df = pd.read_csv(csv_path, sep=',')

    # Use proper pd datatypes
    df['start_time'] = df['start_time'].str.replace('s', '')
    df['start_time'] = df['start_time'].astype(float)
    df['sent_end_time'] = df['sent_end_time'].str.replace('s', '')
    df['sent_end_time'] = df['sent_end_time'].astype(float)
    filename_list = df['filename'].tolist()
    json_name_list = [jname.split('/')[-1] for jname in filename_list]
    audio_name_list = [aname.rsplit('.', 1)[0] + '.ogg' for aname in json_name_list]
    df['audio_name'] = audio_name_list

    return df


def extract_segments(df, ogg_files, excerpt_output_path):
    """
    :param df: Dataframe containing information about the questions.
    :param ogg_files: A list of paths containing only the ogg file locations.
    :param excerpt_output_path: Output path for audio segments extracted.
    :return:
    """
    starts = df.start_time.astype(float)
    ends = df.sent_end_time.astype(float)
    audio_names = df.audio_name.astype(str)

    # slice the audio into segments
    for start, end, audio_name in zip(starts, ends, audio_names):
        # Sub folder list
        ogg_file_sub_list = [jname.split('/')[-3] for jname in ogg_files]
        # Show folder list
        ogg_file_up_list = [jname.split('/')[-2] for jname in ogg_files]
        # Ogg name list
        ogg_file_list = [jname.split('/')[-1] for jname in ogg_files]
        if audio_name in ogg_file_list:
            ogg_index = ogg_file_list.index(audio_name)
            audio_file_path = os.path.join(
                audio_path,
                str(ogg_file_sub_list[ogg_index]),
                str(ogg_file_up_list[ogg_index]),
                audio_name,
            )
            audio_excerpt_name = os.path.join(
                excerpt_output_path,
                str(ogg_file_sub_list[ogg_index]),
                str(ogg_file_up_list[ogg_file_list.index(audio_name)]),
                str(audio_name[:-4] + str(start)),
            )
            print('original path', audio_file_path)
            print('excerpt name', audio_excerpt_name)
            # working in milliseconds
            start = start * 1000
            end = end * 1000
            newAudio = AudioSegment.from_ogg(audio_file_path)
            newAudio = newAudio[start:end]
            newAudio.export(audio_excerpt_name + '.ogg', format='ogg')


def extract_all_audios(csv_path, audio_path, excerpt_output_path):
    """
    :param csv_path: Path of the csv file containing information about
    the questions.
    :param audio_path: A list of paths of all original audio files.
    :param excerpt_output_path: Output path for audio segments extracted.
    """
    df = read_data(csv_path)
    # ogg_files = [f for f in os.listdir(audio_path) if
    #              os.path.isfile(os.path.join(audio_path, f)) and f.endswith('.ogg')]
    ogg_files = extract_all_audio_path(audio_path)
    extract_segments(df, ogg_files, excerpt_output_path)


def extract_all_audio_path(audio_folder_series_path):
    """
    :param audio_folder_series_path: The top level, 0;
                                    Mid-level, 0-5;
                                    Bottom-level, show_00,...
                                    Down-level, 73Kvk... .ogg
    :return: A list of paths containing only the ogg file locations.
    """
    audio_path_list = []
    for top_folder_path in audio_folder_series_path:
        middle_folder_path_list = [x for x in next(os.walk(top_folder_path))[1]]
        for middle_folder_path in middle_folder_path_list:
            bottom_folder_path_list = [
                x
                for x in next(
                    os.walk(os.path.join(top_folder_path, middle_folder_path))
                )[1]
            ]
            for bottom_folder_path in bottom_folder_path_list:
                for filename in os.listdir(
                    os.path.join(
                        top_folder_path, middle_folder_path, bottom_folder_path
                    )
                ):
                    if filename.endswith('ogg'):
                        audio_path_list.append(
                            os.path.join(
                                top_folder_path,
                                middle_folder_path,
                                bottom_folder_path,
                                filename,
                            )
                        )
    return audio_path_list


extract_all_audios(csv_path, [audio_path], excerpt_output_path)
