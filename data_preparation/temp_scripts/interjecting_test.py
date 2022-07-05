from ..containers import *
import json
import os
import numpy as np
import pandas as pd


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
def json_extract_all_sentences(file_name):
    """
    :param file_name: The path of the json files to be extracted.
    :return: Clean string containing the raw sentences.
    """
    # encoding='utf-8', errors='ignore'
    sentence_string = ""
    with open(file_name, "r", errors="ignore") as file_in:
        # Reading from file
        try:
            data = json.loads(file_in.read(), strict=False)
            article_object = Article(data)
            if article_object:
                sentence_list = article_sentence(article_object)
                for each_sentence in sentence_list:
                    sentence_string = sentence_string + str(each_sentence)
        except:
            pass

    return sentence_string


interjecting_sounds_list = [
    "Hmm",
    "eh",
    "oh",
    "ooh",
    "oops",
    "whoa",
    "wow",
    "well",
    "well well well",
]


def get_interjecting_frequency(sentence_string):
    """
    Get the number of interjecting sounds per word.
    :return: A floating number of frequency of interjecting sounds
    """
    count = 0
    transcript_list = sentence_string.split()
    for interjecting_sound in interjecting_sounds_list:
        if interjecting_sound in transcript_list:
            new_count = transcript_list.count(interjecting_sound)
            count += new_count

    if len(transcript_list) > 0:
        interjecting_frequency = count / len(transcript_list)
    else:
        interjecting_frequency = 0.00

    # round to 3sf
    interjecting_frequency = np.format_float_positional(
        interjecting_frequency, unique=False, precision=5
    )
    print(interjecting_frequency)
    return interjecting_frequency


def save_interjecting_frequencies(folder_path_list):
    frequency_list = []
    file_path_list = json_path_in_dir(folder_path_list)
    for json_file in file_path_list:
        sentence_string = json_extract_all_sentences(file_name=json_file)
        interjecting_frequency = get_interjecting_frequency(sentence_string)
        frequency_list.append(interjecting_frequency)

    df = pd.DataFrame(frequency_list, columns=["frequencies"])
    df_name = "interjecting_frequencies_" + str(folder_number) + ".csv"
    save_df_path = os.path.join(home_dir, "data_sheets", df_name)

    print(save_df_path)
    df.to_csv(save_df_path, index=False)


folder_number = 0


home_dir = os.path.join("/home", "yyu")
file_dir = os.path.join(
    home_dir,
    "data",
    "Spotify-Podcasts",
    "podcasts-no-audio-13GB",
    "decompressed-transcripts",
)

app_dir = os.path.join(file_dir, str(folder_number))


save_interjecting_frequencies(folder_path_list=[app_dir])
