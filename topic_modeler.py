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
eng_stopwords = set(stopwords.words('english'))
arabic_stopwords = pd.read_csv('arabic_stopwords.txt', header=None)
arabic_stopwords = arabic_stopwords[0].values.tolist()

# set random seed for reproducable results
np.random.seed(45)

class TopicModeler():
    def __init__(self):
        self.corpus = corpus

    def eng_preprocess(text):
        text = re.sub("[^a-zA-Z]+", " ", text)
        text = word_tokenize(text.lower())
        text = [lemmatizer.lemmatize(x) for x in text]
        text = [x for x in text if x not in eng_stopwords]
        return text

    def arabic_preprocess(text):
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub("\d+", " ", text)
        #s = s.replace('أ','ا').replace('إ','ا').replace('ؤ','ا').replace('ئـ','ا').replace('ء','ا').replace('ة','ه')
        text = ''.join(c for c in text if not ud.category(c).startswith('P'))
        text = ''.join(c for c in text if not ud.category(c).startswith('N'))
        text = text.split(" ")
        text = [x for x in text if x != "" and len(x) > 1]
        text = [j.split("\n") for j in text]
        text = [j for i in text for j in i]
        text = [x for x in text if x not in arabic_stopwords]
        return text

    def apply_tfidf(preprocessed_corpus):
        dictionary = gensim.corpora.Dictionary(preprocessed_corpus)
        dictionary.filter_extremes(no_below = 10, no_above = 0.2)
        bow_corpus = [dictionary.doc2bow(x) for x in preprocessed_corpus]
        tfidf = models.TfidfModel(bow_corpus)
        corpus_tfidf = tfidf[bow_corpus]
        return dictionary, bow_corpus, corpus_tfidf

    def view_doc_tfidf(corpus_tfidf, document_id, n_top_terms, dictionary, chart="n"):
        max_n = heapq.nlargest(n_top_terms, corpus_tfidf[document_id], key=lambda x:x[1])
        print([(dictionary.get(id), value) for id, value in max_n])

        if chart == "y":
            top_words = [(dictionary.get(id), value) for id, value in max_n]
            top_words_dict = dict((x, y) for x, y in top_words)
            plt.bar(range(len(top_words_dict)), top_words_dict.values(), align='center')
            plt.xticks(range(len(top_words_dict)), top_words_dict.keys(), rotation = 90, size=15)
            plt.show()


    def run_lda(bow_corpus, dictionary, passes = 25):
        lda_model = gensim.models.LdaMulticore(bow_corpus,
                                               id2word = dictionary,
                                               passes = passes,
                                               workers = 3)
        for idx, topic in lda_model.print_topics(-1):
            print("Topic: {} \nWords: {}".format(idx, topic))
            print("\n")
        return lda_model

    def print_all_lda_matches(lda_model, bow_corpus, document_num):
        for index, score in sorted(lda_model[bow_corpus[document_num]], key=lambda tup: -1*tup[1]):
            print("\nScore: {}\t \nTopic: {}".format(score, lda_model.print_topic(index, 10)))

    def print_n_lda_matches(lda_model, bow_corpus, document_num, n_top_topics = 3):
        topics_list = [(idx, topic) for idx, topic in lda_model.print_topics(-1)]
        sorted_lda_scores = sorted(lda_model[bow_corpus[document_num]], key=lambda tup: -1*tup[1])
        max_n = heapq.nlargest(n_top_topics, sorted_lda_scores, key=lambda x:x[1])
        for index, topic_score in max_n:
            print("\nTopic: {}\nMatching score: {}\n{}\n".format(topics_list[index][0], topic_score, topics_list[index][1]))

    def translate_ar_topics(ar_lda):
        translator = Translator()
        for idx, topic in ar_lda.print_topics(-1)[:4]:
            split_topic = topic.split("+")
            print(idx)
            print(split_topic)
            for word_score in split_topic:
                ar_word=word_score[word_score.find("*")+1:]
                print(ar_word)
                print(translator.translate(ar_word, dest='en').text)
            print("\n\n")

    def translated_n_lda_matches(lda_model, bow_corpus, document_num, n_top_topics = 3):
        topics_list = [(idx, topic) for idx, topic in lda_model.print_topics(-1)]
        sorted_lda_scores = sorted(lda_model[bow_corpus[document_num]], key=lambda tup: -1*tup[1])
        max_n = heapq.nlargest(n_top_topics, sorted_lda_scores, key=lambda x:x[1])
        for index, topic_score in max_n:
            print("\nTopic: {}\nMatching score: {}\n{}".format(topics_list[index][0], topic_score, topics_list[index][1]))
            split_topic = topics_list[index][1].split("+")
            for word_score in split_topic:
                ar_word=word_score[word_score.find("*")+1:]
                print(translator.translate(ar_word, dest='en').text)
            print("\n\n")

    def english_tldr(corpus, document_num):
        print(summarize(corpus[document_num]))
