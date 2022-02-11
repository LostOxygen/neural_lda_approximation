"""main file to run the lda matching"""
# -*- coding: utf-8 -*-
# !/usr/bin/env python3
import time
import socket
from pprint import pprint
import datetime
import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import webdataset as wds
import gensim
from gensim.models import LdaMulticore
from tqdm import tqdm
from matplotlib import pyplot as plt

from utils.words import save_train_data
from utils.network import KLDivLoss

LDA_PATH = "./models/"
DATA_PATH = "./data/wiki_data.tar" # the path of the data on which the lda should be tested
PLOT_PATH = "./plots/"
NUM_TOPICS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


def plot_difference(mdiff: np.array, num_topics: int) -> None:
    """Helper function to plot difference between models.
       :mdiff: the confusion matrix between the two models
       :num_topics: number of topics used to train the two LDA models

       :retrn: None (but saves the figure)
    """
    _, axs = plt.subplots(figsize=(18, 14))
    data = axs.imshow(mdiff, cmap='RdBu_r', origin='lower')
    plt.title(f"Topic Difference Matrix for {num_topics} topics")
    plt.colorbar(data)
    plt.savefig(PLOT_PATH+f"difference_matrix_{num_topics}.png")
    #plt.show()


def train_lda(num_topics: int, path_suffix: str) -> LdaMulticore:
    """helper function for lda training
       :param num_topics: number of topics to train the LDA model
       :param path_suffix: suffix for the save path of the lda model

       :return: trained LDA Model
    """
    dictionary = gensim.corpora.Dictionary.load_from_text("./data/wikipedia_dump/wiki_wordids.txt")
    bow_list = gensim.corpora.MmCorpus("./data/wikipedia_dump/wiki_bow.mm.bz2")

    # compute the lda model
    ldamodel = LdaMulticore(bow_list, num_topics=num_topics,
                            id2word=dictionary,
                            passes=1, workers=os.cpu_count(),
                            eval_every=0)

    save_path = LDA_PATH + "matching_lda_model" + str(path_suffix)
    if not os.path.isdir(LDA_PATH):
        os.mkdir(LDA_PATH)

    try:
        ldamodel.save(save_path)
    except Exception as exception_lda:
        print("[ could not save the lda model ]")
        print(exception_lda)

    return ldamodel


def save_results(diff_tensor: torch.Tensor) -> None:
    """helper function to visualize and save the resulting differences as an graph image
       :param diff_tensor: tensor with tuples of (NUM_TOPICS, difference)

       :return: None
    """
    print("Average differences per topic number: ")
    print(diff_tensor)

    if not os.path.isdir(PLOT_PATH):
        os.mkdir(PLOT_PATH)

    plt.style.use("ggplot")
    fig, axs = plt.subplots()
    idx = list(range(len(diff_tensor)))
    axs.bar(idx, list(diff_tensor.values()), width=0.35, label="Cosine_Similarity")
    axs.set_xticks(idx)
    axs.set_xticklabels(list(diff_tensor.keys()), rotation=85)
    axs.legend()
    axs.set_xlabel("# Topics")
    axs.set_ylabel("Difference-Value")
    fig.tight_layout()

    plt.savefig(PLOT_PATH+"difference_plot.png")
    # plt.show()


def compare_lda_models(lda1: LdaMulticore, lda2: LdaMulticore,
                       num_topics: int) -> torch.Tensor:
    """helper function to calculate the difference between two lda models according to their
       topic-word distribution.
       :param lda1: the first of the two LDA models
       :param lda2: the second LDA models
       :param num_topics: number of topics used to train the two LDA models

       :return: torch.tensor
    """
    # loss_class = KLDivLoss()
    cosine_similarity = nn.CosineSimilarity(dim=0, eps=1e-8)
    dictionary = gensim.corpora.Dictionary.load_from_text("./data/wikipedia_dump/wiki_wordids.txt")

    # compute the confusion matrix between two models
    mdiff, _ = lda1.diff(lda2, distance='hellinger', num_words=50)
    plot_difference(mdiff, num_topics)

    # iterate over every topic and calculate the average difference
    tmp_diff = 0.0
    for topic_id in range(num_topics):
        topics1 = lda1.get_topic_terms(topicid=topic_id, topn=len(dictionary))
        topics2 = lda2.get_topic_terms(topicid=topic_id, topn=len(dictionary))

        # calculate the similarity using the cosine similarity where -1 means the two topic vectors
        # are completely opposite to each other, while 1 means they are completely similar and
        # rectified. A value of 0 means, they are orthogonal to each other
        cos_sim = cosine_similarity(torch.FloatTensor([tuple[0] for tuple in topics1]),
                                    torch.FloatTensor([tuple[0] for tuple in topics2]))
        tmp_diff += cos_sim

    loss = tmp_diff / num_topics
    print(f"--> loss between current two LDAs: {loss}")
    return loss


def main():
    """main function for lda matching"""
    start = time.perf_counter()
    print("\n\n\n"+"#"*55)
    print("## " + str(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p")))
    print(f"## System: {torch.get_num_threads()} CPU cores with "
          f"{os.cpu_count()} threads and "
          f"{torch.cuda.device_count()} GPUs on {socket.gethostname()}")
    print("#"*55)
    print()

    if not os.path.isfile(DATA_PATH):
        print("## no train/test data found .. creating new one")
        # save the lda model data as training data with labels
        save_train_data(freq_id=None)

    # prepare the loaded data
    print("--> Loading Dataset to memory")
    dataset = wds.WebDataset(DATA_PATH).decode().shuffle(1000).to_tuple("input.pyd",
                                                                        "output.pyd")
    loader = DataLoader((dataset.batched(1)), batch_size=None, num_workers=0)
    # get the first bow from the dataloader
    _, bow = next(enumerate(loader))

    # convert sparse tensor back into dense form
    bow = bow[0].to_dense()

    # convert tensor back into bag of words list for the lda model
    bow = bow[0].tolist()
    bow = [(id, int(counting)) for id, counting in enumerate(bow)]

    # dictionary for the CE results in the format: {NUM_TOPIC : avg. CE value}
    avg_results = dict()

    for curr_num_topics in NUM_TOPICS:
        print(f"--> Current number of topics: {curr_num_topics}")

        # train or load a reference lda for the current topic number
        if os.path.isfile("./models/matching_lda_model_ref_"+str(curr_num_topics)):
            print("--> Loading reference LDA")
            ref_lda = LdaMulticore.load("./models/matching_lda_model_ref_"+str(curr_num_topics))
        else:
            print("--> Training reference LDA")
            ref_lda = train_lda(curr_num_topics, "_ref_"+str(curr_num_topics))

        # train or load LDA_ITERS new lda's to compare their results with CE
        if os.path.isfile("./models/matching_lda_model_tmp_"+str(curr_num_topics)):
            print("--> Loading second LDA")
            tmp_lda = LdaMulticore.load("./models/matching_lda_model_tmp_"+str(curr_num_topics))
        else:
            print("--> Training second LDA")
            tmp_lda = train_lda(curr_num_topics, "_tmp_"+str(curr_num_topics))

        # compare the two LDA models and build the result matrix
        curr_lda_diff = compare_lda_models(ref_lda, tmp_lda, curr_num_topics)
        avg_results[curr_num_topics] = curr_lda_diff # add to the dictionary

    save_results(avg_results)

    end = time.perf_counter()
    duration = (np.round(end - start) / 60.) / 60.
    print(f"\nComputation time: {duration:0.4f} hours")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(**vars(args))
