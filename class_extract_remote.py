"""Extract questions from json transcripts.
This script finds the questions from json transcripts and gather all
information in a csv file.
"""


import json
import os
import pandas as pd
import numpy as np
from containers import *


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


def extract_timings(file_name):
    """
    :param file_name: Absolute file path of json file.
    :return: :Pandas dataframe containing the start times, end times and sentences.
    """
    start_time_list = []
    end_time_list = []
    sent_end_time_list = []
    sentence_string_list = []
    article_object = json_extract(file_name)
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
            # filter out questions which are too short
            mask = questions_df["sentence"].astype(str).str.len() > 30
            questions_df = questions_df.loc[mask]
        return questions_df


def complete_dataframe(folder_path_list):
    """
    :param folder_path_list: Absolute path of parent parent folder.
    :return: Complete pandas dataframe containing the start times, end times and sentences.
    """
    file_path_list = json_path_in_dir(folder_path_list)
    small_dfs = []
    for json_file in file_path_list:
        questions_df = extract_timings(file_name=json_file)
        small_dfs.append(questions_df)
    total_df = pd.concat(small_dfs, ignore_index=True)
    # save_df_path = (
    #     os.path.join(home_dir, 'confidence_dataframe_1') + '.csv'
    # )
    save_df_path = os.path.join(
        home_dir, "data_sheets", "confidence_dataframes", "confidence_dataframe_3.csv"
    )

    print(save_df_path)
    total_df.to_csv(save_df_path, index=False)
    return total_df


# home_dir is the location of person
home_dir = os.path.join("/home", "yyu")
# file_dir = os.path.join(home_dir, 'confidence_css')
file_dir = os.path.join(
    home_dir,
    "data",
    "Spotify-Podcasts",
    "podcasts-no-audio-13GB",
    "decompressed-transcripts",
)

app_dir = os.path.join(file_dir, "3")

complete_dataframe(folder_path_list=[app_dir])
