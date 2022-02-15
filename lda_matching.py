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


def save_results(similarity_tensor: torch.Tensor, diff_tensor: torch.Tensor) -> None:
    """helper function to visualize and save the resulting differences as an graph image
       :param similarity_tensor: tensor with tuples of (NUM_TOPICS, similarites) -> cosine sim.
       :param diff_tensor: tensor with tuples of (NUM_TOPICS, difference) -> KLDiv

       :return: None
    """
    print("Average similarities per topic number: ")
    print(similarity_tensor)
    print("Average differences per topic number: ")
    print(diff_tensor)

    if not os.path.isdir(PLOT_PATH):
        os.mkdir(PLOT_PATH)

    plt.style.use("ggplot")
    fig, axs = plt.subplots()
    idx = np.arange(len(diff_tensor))
    width = 0.35
    axs.bar(idx - width/2, list(diff_tensor.values()), width=width, label="KLDiv")
    axs.bar(idx + width/2, list(similarity_tensor.values()), width=width, label="Cosine_Similarity")
    axs.set_xticks(idx)
    axs.set_yticks(np.arange(0., 1.1, 0.1))
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
    print("--> Compare LDA models")
    kldiv_loss = KLDivLoss()
    cosine_similarity = nn.CosineSimilarity(dim=0, eps=1e-8)
    dictionary = gensim.corpora.Dictionary.load_from_text("./data/wikipedia_dump/wiki_wordids.txt")

    # compute the confusion matrix between two models
    mdiff, _ = lda1.diff(lda2, distance='hellinger', num_words=50)
    plot_difference(mdiff, num_topics)

    # iterate over every topic and calculate the average difference
    tmp_sim = 0.0 # temp var for cosine similarity
    tmp_diff = 0.0 # temp var for kullback leibler divergence
    for topic_id in range(num_topics):
        topics1 = lda1.get_topic_terms(topicid=topic_id, topn=len(dictionary))
        topics2 = lda2.get_topic_terms(topicid=topic_id, topn=len(dictionary))

        # vectors with the word IDs in their contribution order for the current topic ID
        word_id_vec1 = [tuple[0] for tuple in topics1]
        word_id_vec2 = [tuple[0] for tuple in topics2]

        # calculate the similarity using the cosine similarity where -1 means the two topic vectors
        # are completely opposite to each other, while 1 means they are completely similar and
        # rectified. A value of 0 means, they are orthogonal to each other
        cos_sim = cosine_similarity(torch.FloatTensor(word_id_vec1),
                                    torch.FloatTensor(word_id_vec2))

        # since cos_sim is the distance between the vectors, (1 - cos_sim) is the similarity
        tmp_sim += (1- cos_sim)

        # empty vectors for the word probabilities to calculate their difference
        word_prob_vec1 = torch.zeros(len(dictionary))
        word_prob_vec2 = torch.zeros(len(dictionary))

        for (word_tuple1, word_tuple2) in zip(topics1, topics2):
            # assign the word probabilites to their ID in the vector
            word_prob_vec1[word_tuple1[0]] = torch.FloatTensor([word_tuple1[1]])
            word_prob_vec2[word_tuple2[0]] = torch.FloatTensor([word_tuple2[1]])

            tmp_diff += kldiv_loss(word_prob_vec1, word_prob_vec2)

    difference = tmp_diff / num_topics
    similarity = tmp_sim / num_topics
    print(f"--> similarity between current two LDA word vectors: {similarity}")
    print(f"--> difference between current two LDA prob. distributions: {difference}")
    return similarity, difference


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

    # dictionary for the similarity results in the format: {NUM_TOPIC : similarity}
    similarity_results = dict() # similarity between the word vectors and their order
    difference_results = dict() # difference between the probability distributions

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
        curr_lda_sim, curr_lda_diff = compare_lda_models(ref_lda, tmp_lda, curr_num_topics)
        # add to the dictionary
        similarity_results[curr_num_topics] = curr_lda_sim
        difference_results[curr_num_topics] = curr_lda_diff

    save_results(similarity_results, difference_results)

    end = time.perf_counter()
    duration = (np.round(end - start) / 60.) / 60.
    print(f"\nComputation time: {duration:0.4f} hours")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(**vars(args))
