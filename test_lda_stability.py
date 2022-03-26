"""main file to run the lda stability test"""
# -*- coding: utf-8 -*-
# !/usr/bin/env python3
import time
import socket
import datetime
import argparse
import os
from pprint import pprint
import torch
import numpy as np
import gensim
from gensim.models import LdaMulticore
from gensim.corpora import Dictionary, MmCorpus, IndexedCorpus
from gensim.test.utils import get_tmpfile
from gensim.utils import ClippedCorpus
from matplotlib import pyplot as plt

LDA_PATH = "./models/" # the path of the data on which the lda should be tested
DATA_PATH = "./data/wiki_data.tar"
PLOT_PATH = "./plots/"
NUM_ARTICLES = 10000 # number of articles to be used to calc. the mean article intersections
NUM_TOPICS = 20 # number of topics the LDA is trained on
NUM_TOP_ARTICLES = 5  # number of top N articles to be used for the article intersections


def train_lda(num_topics: int, path_suffix: str, train_size: float,
              bow_corpus: MmCorpus, dictionary: Dictionary) -> LdaMulticore:
    """helper function for lda training
       :param num_topics: number of topics to train the LDA model
       :param path_suffix: suffix for the save path of the lda model
       :param train_size: percentage size of the corpus to use for lda training

       :return: trained LDA Model
    """
    # clip the train size to prevent the percentage to be under 0.01 or over 1.0
    train_size = np.clip(train_size, 0.01, 1.)
    # create a cropus wrapper which ends at the desired train_size percentage of the normal corpus
    bow_clipped = ClippedCorpus(bow_corpus, max_docs=int(len(bow_corpus) * train_size))

    # compute the lda model
    ldamodel = LdaMulticore(bow_clipped, num_topics=num_topics,
                            id2word=dictionary,
                            passes=1, workers=os.cpu_count(),
                            eval_every=0)

    save_path = LDA_PATH + "stability_lda_model" + str(path_suffix)
    if not os.path.isdir(LDA_PATH):
        os.mkdir(LDA_PATH)

    try:
        ldamodel.save(save_path)
    except Exception as exception_lda:
        print("[ could not save the lda model ]")
        print(exception_lda)

    return ldamodel


def get_similarity(topics_i: list, topics_j: list) -> float:
    """Helper function which calculates and returns the ranking based on the
       number of intersections between the two topic lists.
       :param topics_i: list of topics of the first document
       :param topics_j: list of topics of the second document
       :return: the similarity of the two topic vectors
    """
    topics_i = torch.tensor(topics_i)
    topics_j = torch.tensor(topics_j)

    return torch.dot(topics_i, topics_j.T)


def visualize_results(stability_dict: dict) -> None:
    """Helper function to plot the stability values for the corresponding corpus sizes
       :param stability_dict: dictionary of the stability values
       :return: None
    """
    print("Stability Values:")
    pprint(stability_dict)

    if not os.path.isdir(PLOT_PATH):
        os.mkdir(PLOT_PATH)

    plt.style.use("ggplot")
    fig, axs = plt.subplots()
    idx = np.arange(len(stability_dict))
    width = 0.35
    rects1 = axs.bar(idx - width/2, list(stability_dict.values()), width=width,
                     label="Num_Inters.")
    axs.set_xticks(idx)
    # use the biggest value of the difference dictionary as the maximum reference for y-axis
    axs.set_yticks(np.arange(0., stability_dict[max(stability_dict, key=stability_dict.get)], 0.5))
    axs.set_xticklabels(list(stability_dict.keys()), rotation=85)

    for rect in rects1:
        height = rect.get_height()
        axs.annotate(f"{np.round(height, 2)} / {float(NUM_TOP_ARTICLES)}",
                     xy=(rect.get_x() + rect.get_width() / 2, height),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords="offset points",
                     ha="center", va="bottom")

    axs.legend()
    axs.set_xlabel("Corpus Size (in %)")
    axs.set_ylabel(f"Mean Article Intersections")
    fig.tight_layout()

    plt.savefig(PLOT_PATH+"stability_plot.png")
    plt.show()


def get_lda_stability(lda1: LdaMulticore, lda2: LdaMulticore, articles_i: list,
                      articles_j: list) -> float:
    """Helper function to measure the similarity of two documents by using their
       topic intersection length.
       :param lda1: first LDA model
       :param lda2: second LDA model
       :param articles_i: first list of random articles
       :param articles_j: second list of random articles

       :return: stability difference between the two LDAs w.r.t to the articles
    """
    # obtain the topic lists of the documents and save them in a list
    # also only keep their topic ID and cut of the probabilities
    # (topic embedding of the articles)
    lda1_article_topics_i = [topic_tuple[0] for topic_tuple in
                             [lda1.get_document_topics(article)
                              for article in articles_i]]
    lda1_article_topics_j = [topic_tuple[0] for topic_tuple in
                             [lda1.get_document_topics(article)
                              for article in articles_j]]

    lda2_article_topics_i = [topic_tuple[0] for topic_tuple in
                             [lda2.get_document_topics(article)
                              for article in articles_i]]
    lda2_article_topics_j = [topic_tuple[0] for topic_tuple in
                             [lda2.get_document_topics(article)
                              for article in articles_j]]

    total_intersection_size = 0.

    # iterate over the topics of every document and compare their topics to the target
    for lda1_topics_j, lda2_topics_j in zip(lda1_article_topics_j, lda2_article_topics_j):

        lda1_document_ranking = []
        lda2_document_ranking = []

        for doc_id, (lda1_topics_i, lda2_topics_i) in enumerate(zip(lda1_article_topics_i,
                                                                    lda2_article_topics_i)):

            # calculate the score between two articles
            lda1_score = get_similarity(lda1_topics_i, lda1_topics_j)
            lda2_score = get_similarity(lda2_topics_i, lda2_topics_j)

            lda1_document_ranking.append((doc_id, lda1_score))
            lda2_document_ranking.append((doc_id, lda2_score))

        # sort the document ids w.r.t their ranking values
        lda1_document_ranking.sort(key=lambda x: x[1])
        lda2_document_ranking.sort(key=lambda x: x[1])

        # create a set of the top 5 documents
        lda1_top5_docs = {doc_tuple[0] for doc_tuple in lda1_document_ranking[:NUM_TOP_ARTICLES]}
        lda2_top5_docs = {doc_tuple[0] for doc_tuple in lda2_document_ranking[:NUM_TOP_ARTICLES]}
        #print(f"LDA1 top 5 documents: {lda1_top5_docs}")
        #print(f"LDA2 top 5 documents: {lda2_top5_docs}")

        # add their intersection size to the total intersection size
        total_intersection_size += len(lda1_top5_docs.intersection(lda2_top5_docs))

    # normalize the total intersection size by the number of documents
    total_intersection_size /= len(lda1_article_topics_j)
    print("Mean Intersection Size: ", total_intersection_size)

    return total_intersection_size


def main(corpus_sizes: float) -> None:
    """main function for lda stability testing"""
    start = time.perf_counter()
    print("\n\n\n"+"#"*55)
    print("## " + str(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p")))
    print(f"## System: {torch.get_num_threads()} CPU cores with "
          f"{os.cpu_count()} threads and "
          f"{torch.cuda.device_count()} GPUs on {socket.gethostname()}")
    print("#"*55)
    print()

    print("Load corpus data..")
    dictionary = Dictionary.load_from_text("./data/wikipedia_dump/wiki_wordids.txt")
    # serialize the corpus to allow O(1) access to the data by indexing
    bow_corpus = MmCorpus(fname="./data/wikipedia_dump/wiki_tfidf.mm")

    # MmCorpus.serialize("./data/wikipedia_dump/wiki_bow.mm", bow_corpus)

    # dictionary with the corpus size as a key and the stability value as the values
    stability_values = {}

    for corpus_size in corpus_sizes:
        print(f"Current corpus size factor: {corpus_size}")
        # check if a lda with the current corpus size already exists
        if os.path.isfile(f"./models/stability_lda_model_A{corpus_size}"):
            print("--> Loading first LDA")
            lda1 = LdaMulticore.load(f"./models/stability_lda_model_A{corpus_size}")
        else:
            print("--> Training first LDA")
            lda1 = train_lda(NUM_TOPICS, f"_A{corpus_size}", corpus_size, bow_corpus, dictionary)

        if os.path.isfile(f"./models/stability_lda_model_B{corpus_size}"):
            print("--> Loading second LDA")
            lda2 = LdaMulticore.load(f"./models/stability_lda_model_B{corpus_size}")
        else:
            print("--> Training second LDA")
            lda2 = train_lda(NUM_TOPICS, f"_B{corpus_size}", corpus_size, bow_corpus, dictionary)

        articles_i = []
        articles_j = []

        # fill the article lists with random articles from the bow_corpus
        for _ in range(NUM_ARTICLES):
            articles_i.append(bow_corpus[np.random.randint(len(bow_corpus))])
            articles_j.append(bow_corpus[np.random.randint(len(bow_corpus))])

        
        stability_values[corpus_size] = get_lda_stability(lda1, lda2, articles_i, articles_j)

    visualize_results(stability_values)

    end = time.perf_counter()
    duration = (np.round(end - start) / 60.) / 60.
    print(f"\nComputation time: {duration:0.4f} hours")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_sizes", "-cs", help="Defines the corpus size in percent",
                        nargs='+', type=float, default=0.1)

    args = parser.parse_args()
    main(**vars(args))
