"""main file to run the lda matching"""
# -*- coding: utf-8 -*-
# !/usr/bin/env python3
import time
import socket
import datetime
import argparse
import os
import torch
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
# number of iterations of lda bow evaluations per topic number to calc. the average CE
LDA_ITERS = 100


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
    axs.bar(idx, list(diff_tensor.values()), width=0.35, label="KLDiv")
    axs.set_xticks(idx)
    axs.set_xticklabels(list(diff_tensor.keys()), rotation=85)
    axs.legend()
    axs.set_xlabel("# Topics")
    axs.set_ylabel("Difference-Value")
    fig.tight_layout()

    plt.savefig(PLOT_PATH+"difference_plot.png")
    # plt.show()


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

    # dictionary for the CE results in the format: {NUM_TOPIC : avg. CE value}
    avg_ce_results = dict()
    loss_class = KLDivLoss()

    for curr_num_topics in NUM_TOPICS:
        print(f"--> Current number of topics: {curr_num_topics}")
        temp_avg_ce_value = 0.0

        # train a reference lda for the current topic number
        print("--> Training reference LDA")
        ref_lda = train_lda(curr_num_topics, "_ref_"+str(curr_num_topics))
        ref_lda_output = ref_lda.get_document_topics(bow)

        # train LDA_ITERS new lda's to compare their results with CE
        print("--> Training second LDA")
        tmp_lda = train_lda(curr_num_topics, "_tmp_"+str(curr_num_topics))
        tmp_lda_output = tmp_lda.get_document_topics(bow)

        # iterate over LDA_ITERS bag of words to calculat the average difference
        for _ in tqdm(range(LDA_ITERS)):
            # get a new bow from the dataloader
            _, bow = next(enumerate(loader))

            # convert sparse tensor back into dense form
            bow = bow[0].to_dense()

            # convert tensor back into bag of words list for the lda model
            bow = bow[0].tolist()
            bow = [(id, int(counting)) for id, counting in enumerate(bow)]

            # create an empty vector with CURR_NUM_ZEROS elements to insert the probs
            # of the corresponding topic id (so they have always the same size)
            ref_lda_vec = torch.zeros(curr_num_topics) + 1e-10
            for bow_tuple in ref_lda_output:
                ref_lda_vec[bow_tuple[0]] = torch.tensor(bow_tuple[1])

            # create an empty vector with CURR_NUM_ZEROS elements to insert the probs
            # of the corresponding topic id (so they have always the same size)
            tmp_lda_vec = torch.zeros(curr_num_topics) + 1e-10
            for bow_tuple in tmp_lda_output:
                tmp_lda_vec[bow_tuple[0]] = torch.tensor(bow_tuple[1])

            loss = loss_class(ref_lda_vec, tmp_lda_vec)
            temp_avg_ce_value += loss
            print(f"--> loss between current lda and reference: {loss}")

        temp_avg_ce_value /= LDA_ITERS # calculate the average CE value
        avg_ce_results[curr_num_topics] = temp_avg_ce_value # add to the dictionary

    save_results(avg_ce_results)

    end = time.perf_counter()
    duration = (np.round(end - start) / 60.) / 60.
    print(f"\nComputation time: {duration:0.4f} hours")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(**vars(args))
