"""main file to run the lda matching"""
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import time
import argparse
import os
import numpy as np
import gensim
from gensim.models import LdaMulticore

LDA_PATH = "./models/"
DATA_PATH = "./data/"

def train_lda(num_topics: int, path_suffix: str) -> gensim.models.LdaMulticore:
    """helper function for lda training
       :param num_topics: number of topics for the lda
       :param path_suffix: suffix for the save path of the lda model

       :return: trained LDA Model
    """
    dictionary = gensim.corpora.Dictionary.load_from_text("./data/wikipedia_dump/wiki_wordids.txt")
    bow_list = gensim.corpora.MmCorpus("./data/wikipedia_dump/wiki_bow.mm.bz2")

    # compute the lda model
    ldamodel = LdaMulticore(bow_list, num_topics=num_topics,
                            id2word=dictionary,
                            passes=2, workers=-1,
                            eval_every=0)

    save_path = "matching_lda_model" + str(path_suffix)
    print("[ saving lda model in {} ]".format(save_path))
    if not os.path.isdir(LDA_PATH):
        os.mkdir(LDA_PATH)

    try:
        ldamodel.save(save_path)
    except Exception as exception_lda:
        print("[ could not save the lda model ]")
        print(exception_lda)

    return ldamodel


def main():
    """main function for lda matching"""
    start = time.perf_counter()

    # TODO: LDA MATCHING IMPLEMENTIEREN

    end = time.perf_counter()
    duration = (np.round(end - start) / 60.) / 60.
    print(f"\nComputation time: {duration:0.4f} hours")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(**vars(args))
