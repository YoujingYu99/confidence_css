"""Utility functions for extracting questions from transcript files and
extract audio segments from their parent audio files.
"""

import json
import os
import pandas as pd
import numpy as np
from containers import *
from pydub import AudioSegment
from pydub.utils import which
import xml.etree.ElementTree as ET

AudioSegment.converter = which("ffmpeg")


def read_data(csv_path):
    """
    :param csv_path: Path of the csv file containing information about
    the questions.
    :return: A dataframe containing 'start_time', 'sent_end_time' and
    'filename'.
    """
    df = pd.read_csv(csv_path, sep=",")

    # Use proper pd datatypes
    df["start_time"] = df["start_time"].str.replace("s", "")
    df["start_time"] = df["start_time"].astype(float)
    df["sent_end_time"] = df["sent_end_time"].str.replace("s", "")
    df["sent_end_time"] = df["sent_end_time"].astype(float)
    filename_list = df["filename"].tolist()
    json_name_list = [jname.split("/")[-1] for jname in filename_list]
    audio_name_list = [aname.rsplit(".", 1)[0] + ".ogg" for aname in json_name_list]
    df["audio_name"] = audio_name_list

    return df


def extract_segments(df, ogg_files, audio_path, excerpt_output_path):
    """
    :param df: Dataframe containing information about the questions.
    :param ogg_files: A list of paths containing only the ogg file locations.
    :param audio_path: A path of the folder.
    :param excerpt_output_path: Output path for audio segments extracted.
    :return:
    """
    starts = df.start_time.astype(float)
    ends = df.sent_end_time.astype(float)
    audio_names = df.audio_name.astype(str)

    # slice the audio into segments
    for start, end, audio_name in zip(starts, ends, audio_names):
        # Sub folder list
        ogg_file_sub_list = [jname.split("/")[-3] for jname in ogg_files]
        # Show folder list
        ogg_file_up_list = [jname.split("/")[-2] for jname in ogg_files]
        # Ogg name list
        ogg_file_list = [jname.split("/")[-1] for jname in ogg_files]
        if audio_name in ogg_file_list:
            ogg_index = ogg_file_list.index(audio_name)
            audio_file_path = os.path.join(
                audio_path,
                str(ogg_file_sub_list[ogg_index]),
                str(ogg_file_up_list[ogg_index]),
                audio_name,
            )
            audio_excerpt_name = (
                str(ogg_file_sub_list[ogg_index])
                + "_"
                + str(ogg_file_up_list[ogg_file_list.index(audio_name)])
                + "_"
                + str(audio_name[:-4] + "_" + str(start))
            )
            audio_excerpt_name.replace("/", "_")
            print(audio_excerpt_name)
            audio_excerpt_path_name = os.path.join(
                excerpt_output_path, audio_excerpt_name
            )
            print(audio_excerpt_path_name)
            # working in milliseconds
            start = start * 1000
            end = end * 1000
            try:
                newAudio = AudioSegment.from_ogg(audio_file_path)
                newAudio = newAudio[start:end]
                newAudio.export(audio_excerpt_path_name + ".mp3", format="mp3")
            except:
                pass


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
    extract_segments(df, ogg_files, audio_path[0], excerpt_output_path)
    print("Extraction finished!")


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
                    if filename.endswith("ogg"):
                        audio_path_list.append(
                            os.path.join(
                                top_folder_path,
                                middle_folder_path,
                                bottom_folder_path,
                                filename,
                            )
                        )
    return audio_path_list


def json_path_in_dir(folder_path_list):
    """
    :folder_path_list: The path of the parent-parent folder of json
                        top: 0
                        middle: 0, 1, ... , 9, A, B, .., Z
                        bottom:show_002B8
    :return: Clean list containing the absolute filepaths of all json files.
    """
    file_path_list = []
    for top_folder_path in folder_path_list:
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
                    if filename.endswith("json"):
                        file_path_list.append(
                            os.path.join(
                                top_folder_path,
                                middle_folder_path,
                                bottom_folder_path,
                                filename,
                            )
                        )
    return file_path_list


# open JSON file
def json_extract(file_name):
    """
    :param file_name: The path of the json files to be extracted.
    :return: Clean list containing the raw sentences.
    """
    # encoding='utf-8', errors='ignore'
    with open(file_name, "r", errors="ignore") as file_in:
        # Reading from file
        try:
            data = json.loads(file_in.read(), strict=False)
            return Article(data)
        except:
            pass


def article_sentence(article_object):
    """
    :param article_object: The article object to be read.
    :return: Clean list containing the raw sentences.
    """
    if article_object.sentence_list:
        sentence_list = article_object.sentence_list
        return sentence_list
    else:
        pass


def sentence_word(sentence_object):
    """
    :param Sentence_object: The sentence object to be read.
    :return: Clean list containing the dictionaries of individual words.
    """
    word_list = sentence_object.word_list
    return word_list


def find_last_word(before_word_list):
    """
    :param before_word_list: The list of words before.
    :return: The last word.
    """
    end_word_list = []
    for word_dict in before_word_list:
        if (
            "." in str(word_dict.word)
            or "?" in str(word_dict.word)
            or "!" in str(word_dict.word)
        ):
            end_word_list.append(word_dict)
    if len(end_word_list) == 0:
        first = {"startTime": "0.000s", "endTime": None, "word": None}
        first = Word(first)
    else:
        first = end_word_list[-1]
    return first


def from_timings_extract_transcript(word_list, start_time, end_time):
    """
    :param word_list: A list of word dictionaries.
    :param start_time: Start time of sentence.
    :param end_time: End time of sentence.
    :return: Sentence string.
    """
    sentence_list = []
    for word_dict in word_list:
        if float(word_dict.startTime[:-1]) >= float(start_time[:-1]) and float(
            word_dict.endTime[:-1]
        ) <= float(end_time[:-1]):
            sentence_list.append(word_dict.word)
    sentence_string = " ".join(sentence_list[1:])
    return sentence_string


def get_show_category(rss_folder_dir, file_name):
    """
    Get the category of the show from the rss data.
    :param rss_folder_dir: Top folder of rss.
    :param file_name: Full path to the audio file.
    :return: show_category: String of category of the show
    """
    # top folder
    ogg_file_top = file_name.split("/")[-4]
    # sub folder list
    ogg_file_sub = file_name.split("/")[-3]
    # show name list
    show_file = file_name.split("/")[-2] + ".xml"
    xml_path = os.path.join(rss_folder_dir, ogg_file_top, ogg_file_sub, show_file)

    # Parse xml file
    tree = ET.parse(xml_path)
    root = tree.getroot()
    # Get the category
    show_category = str(
        root.find("./channel/{http://www.itunes.com/dtds/podcast-1.0.dtd}category").get(
            "text"
        )
    )
    print(show_category)

    return show_category


def extract_timings(rss_folder_dir, file_name):
    """
    :param rss_folder_dir: Absolute folder path of rss files.
    :param file_name: Absolute file path of json file.
    :return: :Pandas dataframe containing the start times, end times and sentences.
    """
    start_time_list = []
    end_time_list = []
    sent_end_time_list = []
    sentence_string_list = []
    article_object = json_extract(file_name)
    show_category = get_show_category(rss_folder_dir, file_name)
    if article_object:
        sentence_list = article_sentence(article_object)
        for sentence in sentence_list:
            word_list = sentence_word(sentence)
            # ignore wordlist if it is none
            if isinstance(word_list, type(None)):
                continue
            else:
                for word_dic in word_list:
                    if word_dic.speakerTag:
                        # print('multiple speakers')
                        if "?" in str(word_dic.word):
                            # print('questions found')
                            end_time = word_dic.endTime
                            # find the start time of the word spoken next and use it as the end time of the sentence
                            word_dic_index = word_list.index(word_dic)
                            try:
                                sent_end_time = word_list[word_dic_index + 1].startTime
                                last_word_index = next(
                                    (
                                        index
                                        for (index, d) in enumerate(word_list)
                                        if d.word == word_dic.word
                                        and d.endTime == end_time
                                    ),
                                    None,
                                )
                                before_word_list = word_list[:last_word_index]
                                last_word = find_last_word(before_word_list)
                                start_time = last_word.startTime
                                sentence_string = from_timings_extract_transcript(
                                    word_list, start_time, end_time
                                )
                                print(sentence_string)
                                start_time_list.append(start_time)
                                end_time_list.append(end_time)
                                sent_end_time_list.append(sent_end_time)
                                sentence_string_list.append(sentence_string)
                            except:
                                continue
            questions_df = pd.DataFrame(
                np.column_stack(
                    [
                        start_time_list,
                        end_time_list,
                        sent_end_time_list,
                        sentence_string_list,
                    ]
                ),
                columns=["start_time", "end_time", "sent_end_time", "sentence"],
            )
            questions_df["filename"] = file_name
            questions_df["category"] = show_category
            # filter out questions which are too short
            mask = questions_df["sentence"].astype(str).str.len() > 30
            questions_df = questions_df.loc[mask]
        return questions_df


def extract_complete_dataframe(home_dir, folder_number, folder_path_list):
    """
    :param home_dir: Home directory
    :param folder_number: Folder number.
    :param folder_path_list: Absolute path of parent parent folder.
    :return: Complete pandas dataframe containing the start times, end times and sentences.
    """
    rss_folder_dir = os.path.join(
        home_dir, "data", "Spotify-Podcasts", "podcasts-no-audio-13GB", "show-rss",
    )
    file_path_list = json_path_in_dir(folder_path_list)
    small_dfs = []
    for json_file in file_path_list:
        questions_df = extract_timings(rss_folder_dir, file_name=json_file)
        small_dfs.append(questions_df)
    total_df = pd.concat(small_dfs, ignore_index=True)
    dataframe_name = "confidence_dataframe" + str(folder_number) + ".csv"
    save_df_path = os.path.join(
        home_dir, "data_sheets", "confidence_dataframes", dataframe_name
    )

    print(save_df_path)
    total_df.to_csv(save_df_path, index=False)
    return total_df
