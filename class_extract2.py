import json
import os
import pandas as pd
import numpy as np
from containers import *


def json_path_in_dir(folder_path_list):
    """
        :folder_path_list: the path of the parent-parent folder of json
        :return: a clean list containing the absolute filepaths of all json files
        """
    file_path_list = []
    for folder_path in folder_path_list:
        subfolder_list = [x[0] for x in os.walk(folder_path)]
        for subfolder in subfolder_list:
            for filename in os.listdir(os.path.join(folder_path, subfolder)):
                if filename.endswith('json'):
                    file_path_list.append(os.path.join(folder_path, subfolder, filename))
    return file_path_list


# open JSON file
def json_extract(file_name):
    """
    :param file_name: the path of the json files to be extracted
    :return: a clean list containing the raw sentences
    """
    with open(file_name, 'r', encoding='utf-8') as file_in:
        # Reading from file
        data = json.loads(file_in.read())
        return Article(data)


def article_sentence(article_object):
    """
        :param article_object: the article object to be read
        :return: a clean list containing the raw sentences
        """
    sentence_list = article_object.sentence_list
    return sentence_list


def sentence_word(sentence_object):
    """
    :param Sentence_object: the sentence object to be read
    :return: a clean list containing the dictionaries of individual words
    """
    word_list = sentence_object.word_list
    return word_list


def find_last_word(before_word_list):
    end_word_list = []
    for word_dict in before_word_list:
        if '.' in str(word_dict.word) or '?' in str(word_dict.word) or '!' in str(word_dict.word):
            end_word_list.append(word_dict)
    if len(end_word_list) == 0:
        first = {"startTime": "0.000s", "endTime": None, "word": None}
        first = Word(first)
    else:
        first = end_word_list[-1]
    return first


def from_timings_extract_transcript(word_list, start_time, end_time):
    sentence_list = []
    for word_dict in word_list:
        if float(word_dict.startTime[:-1]) >= float(start_time[:-1]) and float(word_dict.endTime[:-1]) <= float(end_time[:-1]):
            sentence_list.append(word_dict.word)
    sentence_string = ' '.join(sentence_list[1:])
    return sentence_string


def extract_timings(file_name):
    """
    :param file_name: absolute file path of json file
    :return: a pandas dataframe containing the start times, end times and sentences
    """
    start_time_list = []
    end_time_list = []
    sent_end_time_list = []
    sentence_string_list = []
    article_object = json_extract(file_name)
    sentence_list = article_sentence(article_object)
    for sentence in sentence_list:
        word_list = sentence_word(sentence)
        # ignore wordlist if it is none
        if isinstance(word_list, type(None)):
            continue
        else:
            for word_dic in word_list:
                if word_dic.speakerTag:
                    #print('multiple speakers')
                    if "?" in str(word_dic.word):
                        #print('questions found')
                        end_time = word_dic.endTime
                        # find the start time of the word spoken next and use it as the end time of the sentence
                        word_dic_index = word_list.index(word_dic)
                        try:
                            sent_end_time = word_list[word_dic_index + 1].startTime
                            last_word_index = next((index for (index, d) in enumerate(word_list) if d.word == word_dic.word and d.endTime == end_time), None)
                            before_word_list = word_list[: last_word_index]
                            last_word = find_last_word(before_word_list)
                            start_time = last_word.startTime
                            sentence_string = from_timings_extract_transcript(word_list, start_time, end_time)
                            start_time_list.append(start_time)
                            end_time_list.append(end_time)
                            sent_end_time_list.append(sent_end_time)
                            sentence_string_list.append(sentence_string)
                        except:
                            continue
        questions_df = pd.DataFrame(np.column_stack([start_time_list, end_time_list,sent_end_time_list, sentence_string_list]),
                                    columns=['start_time', 'end_time','sent_end_time', 'sentence'])
        questions_df['filename'] = file_name
        # filter out questions which are too short
        mask = (questions_df['sentence'].astype(str).str.len() > 30)
        questions_df = questions_df.loc[mask]
    return questions_df


def complete_dataframe(folder_path_list):
    """
    :param folder_path_list: absolute path of parent parent folder
    :return: a complete pandas dataframe containing the start times, end times and sentences
    """
    file_path_list = json_path_in_dir(folder_path_list)
    small_dfs = []
    for json_file in file_path_list:
        questions_df = extract_timings(file_name=json_file)
        small_dfs.append(questions_df)
    total_df = pd.concat(small_dfs, ignore_index=True)
    save_df_path = os.path.join(fileDir, 'confidence_css2') + '.csv'
    print(save_df_path)
    total_df.to_csv(save_df_path, index=False)
    return total_df


fileDir = os.path.dirname(os.path.realpath('__file__'))
app_dir = os.path.join(fileDir, '5')

print(complete_dataframe(folder_path_list=[app_dir]))

# filename = r"C:\Users\Youjing Yu\PycharmProjects\confidence_css\5\show_05As0fQbe0p9CgcaJbek8n\2CJtQvWuxtxxgpjqcL72uP.json"
# print(extract_timings(file_name=filename))


# article = json_extract(filename)
# print(article)
