

class Word:
    def __init__(self, word_dictionary):
        self.speakerTag = None
        self.startTime = word_dictionary['startTime']
        self.endTime = word_dictionary['endTime']
        self.word = word_dictionary['word']
        if 'speakerTag' in word_dictionary:
            self.speakerTag = word_dictionary['speakerTag']
            #print(word_dictionary['speakerTag'])


class Sentence:
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
        ret_str = ""
        for sentence in self.sentence_list:
            ret_str = ret_str + sentence.__str__()
        return ret_str



