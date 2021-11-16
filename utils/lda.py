"""helper module to prepare and train the lda model"""
import os
import gensim
from gensim.models import LdaMulticore

LDA_PATH = "./models/"
DATA_PATH = "./data/"

def train_lda(num_workers: int, num_topics: int, words_list: list) -> None:
    """helper method to train and save a lda model in a specified path
       :param num_workers: number of workers to compute the lda
       :param num_topics: number of topics for the lda
       :param words_list: the list of words on which the lda should be computed
    """
    # dictionary = gensim.corpora.Dictionary(words_list)
    dictionary = gensim.corpora.Dictionary.load_from_text("./data/wikipedia_dump/wiki_wordids.txt")
    bow_list = gensim.corpora.MmCorpus("./data/wikipedia_dump/wiki_bow.mm.bz2")
    # create a word corpus
    # bow_list = list(map(lambda x: dictionary.doc2bow(x), words_list))
    ldamodel = LdaMulticore(bow_list, num_topics=num_topics, id2word=dictionary,
                            passes=2, workers=num_workers, eval_every=0)

    print("[ saving lda model in {} ]".format(LDA_PATH+"lda_model"))
    if not os.path.isdir(LDA_PATH):
        os.mkdir(LDA_PATH)

    try:
        ldamodel.save(LDA_PATH+"lda_model")
    except Exception as exception_lda:
        print("[ could not save the lda model ]")
        print(exception_lda)
