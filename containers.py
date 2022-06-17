"""User containers to break down the json file.

This script defines the containers, according to their classes, required to
break down the json file.
-----
Classes
-----
Word: Level 1 container; {'startTime': '6.600s', 'endTime': '6.800s',
    'word': 'warm'}
Sentence: Level 2 container; {'alternatives': [{'transcript': ' I'm joined
    in the studio by my colleague Michelle Masson, who's the safety
    promotion officer responsible for all things helicopter. I don't know
    about you Michelle, but I'm really excited to get started with our
    monthly podcast series.', 'confidence': 0.8316425085067749, 'words':
Article: Level 3 container; 'results': [{'alternatives': ...
"""


class Word:
    """
    Level 1 container of an individual word.
    :returns a dictionary word_dictionary with keys 'startTime',
    'endTime', 'word'; optional 'speakerTag'.
    """

    def __init__(self, word_dictionary):
        self.speakerTag = None
        self.startTime = word_dictionary['startTime']
        self.endTime = word_dictionary['endTime']
        self.word = word_dictionary['word']
        if 'speakerTag' in word_dictionary:
            self.speakerTag = word_dictionary['speakerTag']


class Sentence:
    """
    Level 2 container of a sentence.
    :returns a dictionary sentence_dict with optional  keys 'transcript',
    'confidence', 'word_list'.
    """

    def __init__(self, sentence_dict):
        self.transcript = None
        self.confidence = None
        self.word_list = None
        if 'transcript' in sentence_dict:
            self.transcript = sentence_dict['transcript']
        if 'confidence' in sentence_dict:
            self.confidence = sentence_dict['confidence']
        if 'words' in sentence_dict:
            self.word_list = [Word(word_dict) for word_dict in sentence_dict['words']]

    def __str__(self):
        return str(self.transcript)


class Article:
    """
    Level 3 container of an entire article.
    :returns a dictionary article_dict with keys 'results',
    'sentence_list'.
    """

    def __init__(self, article_dict):
        all_raw_sentence_list = article_dict['results']
        self.sentence_list = []
        for i in range(len(all_raw_sentence_list)):
            alternatives = all_raw_sentence_list[i]['alternatives']
            first_alternative = alternatives[0]
            if len(first_alternative) == 0:
                continue
            self.sentence_list.append(Sentence(first_alternative))

    def __str__(self):
        ret_str = ''
        for sentence in self.sentence_list:
            ret_str = ret_str + sentence.__str__()
        return ret_str
