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

lemmatizer = WordNetLemmatizer()
#eng_stopwords = set(stopwords.words('english'))
#arabic_stopwords = pd.read_csv('arabic_stopwords.txt', header=None)
#arabic_stopwords = arabic_stopwords[0].values.tolist()

# set random seed for reproducable results
np.random.seed(45)

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

if __name__ == "__main__":
    #Establish Class TextExtraction
    root = "/User/chanejackson/Desktop/"
    c = TextExtraction()

    #Run the program to get the text files
    c.getFiles()
