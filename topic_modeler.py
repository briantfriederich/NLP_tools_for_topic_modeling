'''
Creator: Brian Friederich
Date: 04 May 2019
'''

#import packages used
import pandas as pd
import csv
import re
import gensim
from gensim import corpora, models
import nltk
from nltk.corpus import reuters
nltk.download('punkt')
from nltk.tokenize import word_tokenize
import numpy as np
nltk.download('reuters')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import heapq
import string
import unicodedata as ud
import matplotlib.pyplot as plt
from googletrans import Translator
from gensim.summarization.summarizer import summarize

# Set variables for global use
lemmatizer = WordNetLemmatizer()
eng_stopwords = set(stopwords.words('english'))
ar_stopwords_df = pd.read_csv('arabic_stopwords.txt', header=None)
ar_stopwords = ar_stopwords_df[0].values.tolist()
base_stopwords = eng_stopwords.union(ar_stopwords)

# set random seed for reproducable results
np.random.seed(45)

#testing corpus

sample_corpus = pd.read_csv("irony-labeled.csv")['comment_text']


class StopwordsList():
    '''
    This class compiles a stopwords list from existing Arabic and English
    stopwords as well as custom stopwords added by the user.

    METHODS:
        __init__
        add_stopwords
    '''
    def __init__(self, stopwords):
        '''
        INPUTS:
        stopwords (set): base set of stopwords
        '''
        self.stopwords = base_stopwords
        self.extra_stopwords = set()

    def add_stopwords(self, extra_stopwords):
        '''
        This method appends custom stopwords from user onto stopwords list

        INPUTS:
            extra_stopwords (str): extra stopwords comma-delimited

        OUTPUTS:
            stopwords (set): full stopwords list of base and custom words
        '''
        self.extra_stopwords = set(extra_stopwords.split(","))
        self.stopwords = self.stopwords.union(self.extra_stopwords)
        return self.stopwords

class Preprocessor():
    '''
    This class preprocesses a text field for use in later algorithms

    METHODS:
        __init__
        print_textfield
        preprocess
    '''
    def __init__(self, textfield, all_stopwords):
        '''
        INPUTS:
            textfield (str): text unit to be preprocessed
            all_stopwords (set): full stopwords list including custom words
        '''
        self.textfield = textfield
        self.all_stopwords = all_stopwords

    def print_textfield(self):
        '''
        Prints incoming textfield for debugging
        '''
        print(self.textfield)

    def preprocess(self):
        '''
        This method preprocesses the incoming unit of text by removing
        non-alphabetic and non-Arabic characters, lowercasing alphabetic
        characters, tokenizing on spaces, replacing single-letter and empty
        tokens, and removing all stopwords.

        OUTPUTS:
            textfield (list): bag-of-words processed textfield
        '''
        self.textfield = re.sub("[^a-zA-Z\u0621-\u064A]+", " ", self.textfield).lower()
        self.textfield = self.textfield.split(" ")
        self.textfield = [x for x in self.textfield if x != "" and len(x) > 1]
        self.textfield = [x for x in self.textfield if x not in self.all_stopwords]
        return self.textfield


class FullTextPreparation():
    '''
    This class ties the custom stopwords and preprocessing scripts together.

    METHODS:
        __init__
        algorithms_prework

    '''
    def __init__(self, stopwords, corpus):
        '''
        INPUTS:
            stopwords (set): base stopwords list
            corpus (list): list of individual documents' texts
        '''
        self.stopwords = stopwords
        self.corpus = corpus

    def algorithms_prework(self):
        '''
        This method runs the stopword and preprocessing texts in sequence.

        OUTPUTS:
            preprocessed_corpus (list): list of lists, each mega-item is a preprocessed document
        '''
        print("Enter New Stopwords: ")
        new_stopwords = input()
        stoplist = StopwordsList(self.stopwords)
        stoplist = stoplist.add_stopwords(new_stopwords)
        all_stopwords = stoplist
        preprocessed_corpus = [Preprocessor.preprocess(Preprocessor(x, all_stopwords)) for x in self.corpus]
        return preprocessed_corpus

class TfidfModeler():
    '''
    This class applies a Tf-Idf transform to the preprocessed corpus.

    METHODS:
        __init__
        apply_tfidf
        indiv_doc_tfidf
    '''
    def __init__(self, preprocessed_corpus):
        self.corpus = preprocessed_corpus
        self.dictionary = {}
        self.bow_corpus = []
        self.corpus_tfidf = []

    def apply_tfidf(self):
        self.dictionary = gensim.corpora.Dictionary(self.corpus)
        self.dictionary.filter_extremes(no_below = 10, no_above = 0.2)
        self.bow_corpus = [self.dictionary.doc2bow(x) for x in self.corpus]
        tfidf = models.TfidfModel(self.bow_corpus)
        self.corpus_tfidf = tfidf[self.bow_corpus]
        return self.dictionary, self.bow_corpus, self.corpus_tfidf

    def indiv_doc_tfidf(self, document_id, n_top_terms=5):
        max_n = heapq.nlargest(n_top_terms, self.corpus_tfidf[document_id], key=lambda x:x[1])
        finished_tfidf = [(self.dictionary.get(id), value) for id, value in max_n]
        tfidf_topics = " ".join([x[0] for x in finished_tfidf])
        return tfidf_topics

class LDAModeler():
    def __init__(self, preprocessed_corpus):
        self.corpus = preprocessed_corpus
        tfidf_lda = TfidfModeler(self.corpus)
        self.dictionary, self.bow_corpus, self.corp_lda_tfidf = TfidfModeler.apply_tfidf(tfidf_lda)
        self.lda_model = None

    def run_lda(self, passes = 2):
        self.lda_model = gensim.models.LdaMulticore(self.bow_corpus,
                                               id2word = self.dictionary,
                                               passes = passes,
                                               workers = 3)
        def format_lda_topics(string):
            return string[string.find("*")+1:].replace("\"","")

        for idx, topic in self.lda_model.print_topics(-1):
            full_topic = "Topic: {} \nWords: {}".format(idx, topic)
            lda_topic_joined = "".join([format_lda_topics(x) for x in topic.split("+")])
            print("\n")
        return self.lda_model

    def print_all_lda_matches(self, document_num):
        for index, score in sorted(self.lda_model[self.bow_corpus[document_num]], key=lambda tup: -1*tup[1]):
            print("\nScore: {}\t \nTopic: {}".format(score, self.lda_model.print_topic(index, 10)))
  

    def print_some_lda_matches(self, document_num, threshold):
        for index, score in sorted(self.lda_model[self.bow_corpus[document_num]], key=lambda tup: -1*tup[1]):
            if score >= threshold:
                print("\nScore: {}\t \nTopic: {}".format(score, self.lda_model.print_topic(index, 10)))
            else:
                continue

    def print_top_lda_match(self, document_num):
        max_index, max_score = max(self.lda_model[self.bow_corpus[document_num]], key=lambda x:x[1])
        print("\nScore: {}\t \nTopic: {}".format(max_score, self.lda_model.print_topic(max_index, 10)))



'''
    def print_all_lda_matches(self.lda_model, self.bow_corpus, document_num):
        for index, score in sorted(self.lda_model[self.bow_corpus[document_num]], key=lambda tup: -1*tup[1]):
            print("\nScore: {}\t \nTopic: {}".format(score, self.lda_model.print_topic(index, 10)))


    def print_n_lda_matches(self.lda_model, self.bow_corpus, document_num, n_top_topics = 3):
        topics_list = [(idx, topic) for idx, topic in self.lda_model.print_topics(-1)]
        sorted_lda_scores = sorted(self.lda_model[self.bow_corpus[document_num]], key=lambda tup: -1*tup[1])
        max_n = heapq.nlargest(n_top_topics, sorted_lda_scores, key=lambda x:x[1])
        for index, topic_score in max_n:
            print("\nTopic: {}\nMatching score: {}\n{}\n".format(topics_list[index][0], topic_score, topics_list[index][1]))




class EnglishPreprocessor(eng_corpus):
    def __init__(self):
        self.eng_corpus = eng_corpus
        self.eng_stopwords = set(stopwords.words('english'))

    def eng_preprocess(self.eng_corpus):
        self.eng_corpus = re.sub("[^a-zA-Z]+", " ", self.eng_corpus)
        self.eng_corpus = word_tokenize(self.eng_corpus.lower())
        self.eng_corpus = [lemmatizer.lemmatize(x) for x in self.eng_corpus]
        self.eng_corpus = [x for x in self.eng_corpus if x not in eng_stopwords]
        return self.eng_corpus

class ArabicPreprocessor(arabic_corpus):
    def __init__(self):
        self.ar_corpus = arabic_corpus
        self.ar_stopwords_df = pd.read_csv('arabic_stopwords.txt', header=None)
        self.ar_stopwords = self.ar_stopwords_df[0].values.tolist()

    def arabic_preprocess(self.ar_corpus):
        self.ar_corpus = self.ar_corpus.translate(str.maketrans('', '', string.punctuation))
        self.ar_corpus = re.sub("\d+", " ", self.ar_corpus)
        #s = s.replace('أ','ا').replace('إ','ا').replace('ؤ','ا').replace('ئـ','ا').replace('ء','ا').replace('ة','ه')
        self.ar_corpus = ''.join(c for c in self.ar_corpus if not ud.category(c).startswith('P'))
        self.ar_corpus = ''.join(c for c in self.ar_corpus if not ud.category(c).startswith('N'))
        self.ar_corpus = self.ar_corpus.split(" ")
        self.ar_corpus = [x for x in self.ar_corpus if x != "" and len(x) > 1]
        self.ar_corpus = [j.split("\n") for j in self.ar_corpus]
        self.ar_corpus = [j for i in self.ar_corpus for j in i]
        self.ar_corpus = [x for x in self.ar_corpus if x not in self.ar_stopwords]
        return self.ar_corpus

class TfidfModeler(preprocessed_corpus):
    # instantiate class
    def __init__(self):
        self.corpus = preprocessed_corpus
        self.dictionary = {}
        self.bow_corpus = []
        self.corpus_tfidf = []

    def apply_tfidf(self.corpus):
        self.dictionary = gensim.corpora.Dictionary(self.corpus)
        self.dictionary.filter_extremes(no_below = 10, no_above = 0.2)
        self.bow_corpus = [dictionary.doc2bow(x) for x in self.corpus]
        tfidf = models.TfidfModel(self.bow_corpus)
        self.corpus_tfidf = tfidf[self.bow_corpus]
        return self.dictionary, self.bow_corpus, self.corpus_tfidf

    def view_doc_tfidf(self.corpus_tfidf, self.document_id, n_top_terms, dictionary, chart="n"):
        max_n = heapq.nlargest(n_top_terms, corpus_tfidf[document_id], key=lambda x:x[1])
        print([(dictionary.get(id), value) for id, value in max_n])

        if chart == "y":
            top_words = [(dictionary.get(id), value) for id, value in max_n]
            top_words_dict = dict((x, y) for x, y in top_words)
            plt.bar(range(len(top_words_dict)), top_words_dict.values(), align='center')
            plt.xticks(range(len(top_words_dict)), top_words_dict.keys(), rotation = 90, size=15)
            plt.show()

class LDAModeler(dictionary, bow_corpus):
    def __init__(self):
        self.dictionary = dictionary
        self.bow_corpus = bow_corpus
        self.lda_model = None

    def run_lda(bow_corpus, dictionary, passes = 25):
        self.lda_model = gensim.models.LdaMulticore(bow_corpus,
                                               id2word = dictionary,
                                               passes = passes,
                                               workers = 3)
        for idx, topic in self.lda_model.print_topics(-1):
            print("Topic: {} \nWords: {}".format(idx, topic))
            print("\n")
        return self.lda_model

    def print_all_lda_matches(self.lda_model, self.bow_corpus, document_num):
        for index, score in sorted(self.lda_model[self.bow_corpus[document_num]], key=lambda tup: -1*tup[1]):
            print("\nScore: {}\t \nTopic: {}".format(score, self.lda_model.print_topic(index, 10)))

    def print_n_lda_matches(self.lda_model, self.bow_corpus, document_num, n_top_topics = 3):
        topics_list = [(idx, topic) for idx, topic in self.lda_model.print_topics(-1)]
        sorted_lda_scores = sorted(self.lda_model[self.bow_corpus[document_num]], key=lambda tup: -1*tup[1])
        max_n = heapq.nlargest(n_top_topics, sorted_lda_scores, key=lambda x:x[1])
        for index, topic_score in max_n:
            print("\nTopic: {}\nMatching score: {}\n{}\n".format(topics_list[index][0], topic_score, topics_list[index][1]))

    def translate_ar_topics(self.lda_model):
        translator = Translator()
        for idx, topic in self.lda_model.print_topics(-1)[:4]:
            split_topic = topic.split("+")
            print(idx)
            print(split_topic)
            for word_score in split_topic:
                ar_word=word_score[word_score.find("*")+1:]
                print(ar_word)
                print(translator.translate(ar_word, dest='en').text)
            print("\n\n")

    def translated_n_lda_matches(self.lda_model, self.bow_corpus, document_num, n_top_topics = 3):
        topics_list = [(idx, topic) for idx, topic in self.lda_model.print_topics(-1)]
        sorted_lda_scores = sorted(self.lda_model[self.bow_corpus[document_num]], key=lambda tup: -1*tup[1])
        max_n = heapq.nlargest(n_top_topics, sorted_lda_scores, key=lambda x:x[1])
        for index, topic_score in max_n:
            print("\nTopic: {}\nMatching score: {}\n{}".format(topics_list[index][0], topic_score, topics_list[index][1]))
            split_topic = topics_list[index][1].split("+")
            for word_score in split_topic:
                ar_word=word_score[word_score.find("*")+1:]
                print(translator.translate(ar_word, dest='en').text)
            print("\n\n")

class EnglishTLDR(corpus):
    def __init__(self):
        self.corpus = corpus

    def english_tldr(corpus, document_num):
        print(summarize(corpus[document_num]))

'''

if __name__ == "__main__":


    full_text_prep = FullTextPreparation(base_stopwords, sample_corpus)
    prepped_corpus = FullTextPreparation.algorithms_prework(full_text_prep)

    tfidf = TfidfModeler(prepped_corpus)
    TfidfModeler.apply_tfidf(tfidf)

    lda_instance = LDAModeler(prepped_corpus)
    corpus_lda = LDAModeler.run_lda(lda_instance)
    LDAModeler.print_top_lda_match(lda_instance, 18)


