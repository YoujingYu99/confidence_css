# Extract sentence with question marks
# Filter based on length
# Extract start and end time
# Output to a csv file

import os
import json


# open JSON file
def json_extract(transcript_dir):
    """
    :param transcript_dir: the path of the json files to be extracted
    :return: a clean list containing the raw sentences
    """
    timings_list = []
    for root, dirs, files in os.walk(transcript_dir):
        for json_file in files:
            n = 0
            if json_file.endswith((".json")):
                while n < 1:
                    with open(os.path.join(transcript_dir, json_file), 'r',
                              encoding='utf-8') as file_in:
                        # Reading from file
                        data = json.loads(file_in.read())
                        for list_of_sent in data['results']:
                            for sent in list_of_sent['alternatives']:
                                try:
                                    list_sent_words = sorted(sent['words'],
                                                             key=lambda d:
                                                             d['startTime'])
                                    for word_description in list_sent_words:
                                        if 'speakerTag' in word_description:
                                            word = word_description['word']
                                            if '?' in word:
                                                print('question found')
                                                print(word)
                                                last_word_index = next(
                                                    (index for (index, d) in
                                                     enumerate(
                                                         list_sent_words) if
                                                     d["word"] == word),
                                                    None)
                                                before_word_list = list_sent_words[
                                                                   : last_word_index]
                                                end_time = word_description[
                                                    'endTime']
                                                before_word_list_reverse = sorted(
                                                    before_word_list,
                                                    key=lambda d: d[
                                                        'startTime'],
                                                    reverse=True)
                                                last_word = find_last_word(
                                                    before_word_list_reverse)
                                                start_time = last_word[
                                                    'startTime']
                                        else:
                                            continue
                                    timings_list.append(
                                        tuple((start_time, end_time)))
                                except:
                                    pass

                    n += 1
                    # print(timings_list)


spec_chars = ['!', ''','#','%','&',''', '(', ')',
              '*', '+', ',', '-', '.', '/', ':', ';', '<',
              '=', '>', '?', '@', '[', '\\', ']', '^', '_',
              '`', '{', '|', '}', '~', '–']


def find_last_word(before_word_list):
    for before_word in before_word_list:
        if '.' in before_word['word'] or '?' in before_word[
            'word'] or '!' in before_word['word']:
            first = before_word
            break
    else:
        first = None
    return first


fileDir = os.path.dirname(os.path.realpath('__file__'))
transcript_test_dir = os.path.join(fileDir, '5',
                                   'show_05As0fQbe0p9CgcaJbek8n')
json_extract(transcript_dir=transcript_test_dir)
