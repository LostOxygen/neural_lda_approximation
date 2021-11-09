"""helper module to preprocess and return the reuters corpus"""
import re
import json
import os
from string import punctuation, whitespace
import html
import logging
import nltk
import gensim
from nltk.corpus import reuters
from nltk.corpus import stopwords as st
from stop_words import get_stop_words
from gensim.parsing.preprocessing import STOPWORDS

# download reuters corpus
nltk.download('reuters')
# download wordnet for tokenization
nltk.download('wordnet')
# download the nltk stopwords
nltk.download('stopwords')

DATA_PATH = "./data/"
LABEL_PATH = "./data/labels/"
TRAIN_SET = list(filter(lambda x: x.startswith('train'), reuters.fileids()))
TEST_SET = list(filter(lambda x: x.startswith('test'), reuters.fileids()))
STOPWORDS = frozenset(list(STOPWORDS)+get_stop_words('en')+st.words('english'))
WHITE_PUNC_REGEX = re.compile(r"[%s]+" % re.escape(whitespace + punctuation),
                              re.UNICODE)


def save_train_data() -> None:
    """helper function to load the lda model and save corpus as training data"""
    ldamodel = gensim.models.LdaMulticore.load('./models/lda_model')
    dictionary = ldamodel.id2word

    print("[ saving dictionary data in {} ]".format(DATA_PATH))
    if not os.path.isdir(DATA_PATH):
        os.mkdir(DATA_PATH)
    if not os.path.isdir(DATA_PATH+"training"):
        os.mkdir(DATA_PATH+"training")
    if not os.path.isdir(DATA_PATH+"test"):
        os.mkdir(DATA_PATH+"test")

    for i in TRAIN_SET + TEST_SET:
        logging.info(i)
        bow = dictionary.doc2bow(preprocess(reuters.raw(i)))
        with open(os.path.join(DATA_PATH, i), 'w+') as f:
            json.dump(dict(bow), f)


def save_train_labels() -> None:
    """helper function to load the lda model and save corpus as training data"""
    ldamodel = gensim.models.LdaMulticore.load('./models/lda_model')

    if not os.path.isdir(LABEL_PATH):
        os.mkdir(LABEL_PATH)
    if not os.path.isdir(LABEL_PATH+"training"):
        os.mkdir(LABEL_PATH+"training")
    if not os.path.isdir(LABEL_PATH+"test"):
        os.mkdir(LABEL_PATH+"test")

    for i in TRAIN_SET + TEST_SET:
        logging.info(i)
        with open(os.path.join(DATA_PATH, i)) as f:
            bow = json.load(f)
        with open(os.path.join(LABEL_PATH, i), 'w+') as f:
            json.dump(list(
                map(lambda x: float(x[1]), ldamodel.get_document_topics(
                    list(
                        map(
                            lambda x: (int(x[0]), int(x[1])), bow.items()
                            )
                        ), minimum_probability=0.0)
                    )), f)


def preprocess(document_text: list) -> list:
    """helper function, which will:
       1.) Lowercase it all
       2.) Remove HTML Entities
       3.) Split by punctuations to remove them.
       4.) Stem / Lemmaize
       5.) Remove stop words
       6.) Remove unit length words
       7.) Remove numbers
       :param document_text: the raw document text as a list of strings
       :return: the preprocessed words as a list of strings
    """
    lemma = nltk.wordnet.WordNetLemmatizer()
    def is_num(num: int):
        """sub-function to check if something is a number"""
        return not (num.isdigit() or (num[0] == '-' and num[1:].isdigit()))

    preprocessed = list(
        filter(
            # remove all single numbers
            is_num, filter(
                # remove all words containing only a single letter
                lambda x: len(x) > 1, filter(
                    # remove all stopwords by using the the pip package # nltk
                    lambda x: x not in STOPWORDS, map(
                        # lemmatize the document
                        lambda x: lemma.lemmatize(x), re.split(
                            # split the document at every whitespace or by punctuation
                            WHITE_PUNC_REGEX, html.unescape(
                                document_text.lower()
                            )
                        )
                    )
                )
            )
        )
    )
    return preprocessed



def get_word_list(is_train: bool) -> list:
    """helper function to return the preprocessed world list
    :return: the words in form of a list
    """
    word_set = TRAIN_SET if is_train else TEST_SET
    words_list = list(
        map(
            # preprocess the raw text
            preprocess, map(
                # convert the training set into raw text
                lambda x: reuters.raw(x), word_set
            )
        )
    )
    return words_list
