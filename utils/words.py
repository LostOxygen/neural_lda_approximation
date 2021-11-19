"""helper module to preprocess and return the reuters corpus"""
import re
import json
import os
from string import punctuation, whitespace
import html
import logging
import nltk
import gensim
import torch
from tqdm import tqdm
from nltk.corpus import reuters
from nltk.corpus import stopwords as st
from stop_words import get_stop_words
from gensim.parsing.preprocessing import STOPWORDS
import webdataset as wds

DATA_PATH = "./data/"
LABEL_PATH = "./data/labels/"
TRAIN_SET = list(filter(lambda x: x.startswith('train'), reuters.fileids()))
TEST_SET = list(filter(lambda x: x.startswith('test'), reuters.fileids()))
STOPWORDS = frozenset(list(STOPWORDS)+get_stop_words('en')+st.words('english'))
WHITE_PUNC_REGEX = re.compile(r"[%s]+" % re.escape(whitespace + punctuation), re.UNICODE)


def save_train_data() -> None:
    """helper function to load the lda model and save corpus as training data"""
    bow_model_data = gensim.corpora.MmCorpus("./data/wikipedia_dump/wiki_bow.mm.bz2")
    lda_model = gensim.models.LdaMulticore.load('./models/lda_model')
    dictionary = lda_model.id2word


    print("[ saving train data and labels .. ]")
    if not os.path.isdir(DATA_PATH):
        os.mkdir(DATA_PATH)
    if not os.path.isdir(DATA_PATH+"training"):
        os.mkdir(DATA_PATH+"training")
    if not os.path.isdir(DATA_PATH+"test"):
        os.mkdir(DATA_PATH+"test")

    sink = wds.TarWriter(DATA_PATH+"wiki_data.tar")
    # iterate over every document in the model
    for index, doc in tqdm(enumerate(bow_model_data)):
        # create the according document topics from the lda as training targets
        target = [*map(lambda x: float(x[1]),
                       lda_model.get_document_topics(
                           list(map(
                               lambda x: (int(x[0]), int(x[1])), doc
                           )), minimum_probability=0.0))]
        target = torch.FloatTensor(target)

        # create a dictionary for every BoW to access the number of occurrences of every word
        doc_dict = dict(doc)

        # create indizes with the corresponding values to create a sparse matrix
        sparse_indizes = []
        sparse_inputs = []
        for key, val in doc_dict.items():
            sparse_indizes.append(int(key))
            sparse_inputs.append(float(val))

        sparse_indizes = torch.LongTensor(sparse_indizes)
        sparse_inputs = torch.FloatTensor(sparse_inputs)
        # create a sparse tensor out of the indize and value tensors
        input_d = torch.sparse.FloatTensor(sparse_indizes.unsqueeze(0), sparse_inputs,
                                           torch.Size([len(dictionary)]))

        # write everything as python pickles into a tar file
        sink.write({
            "__key__": "sample%06d" % index,
            "input.pyd": input_d,
            "output.pyd": target,
        })
    sink.close()

    print("[ saving test data and labels .. ]")
    if not os.path.isdir(LABEL_PATH):
        os.mkdir(LABEL_PATH)
    if not os.path.isdir(LABEL_PATH+"training"):
        os.mkdir(LABEL_PATH+"training")
    if not os.path.isdir(LABEL_PATH+"test"):
        os.mkdir(LABEL_PATH+"test")

    for i in tqdm(TEST_SET):
        logging.info(i)
        bow = dict(dictionary.doc2bow(preprocess(reuters.raw(i))))
        with open(os.path.join(DATA_PATH, i), 'w+') as file:
            # dump the test data
            json.dump(bow, file)
        with open(os.path.join(LABEL_PATH, i), 'w+') as file:
            # dump the test labels
            json.dump(list(
                map(lambda x: float(x[1]), lda_model.get_document_topics(
                    list(
                        map(
                            lambda x: (int(x[0]), int(x[1])), bow.items()
                            )
                        ), minimum_probability=0.0)
                    )), file)

    del bow_model_data
    del lda_model


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
