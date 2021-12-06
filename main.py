"""main file to run the lda approximation"""
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import time
import socket
import logging
import datetime
import argparse
import os
import torch
import numpy as np
from utils.words import save_train_data
from utils.lda import train_lda
from utils.train import train
from utils.eval import evaluate

torch.backends.cudnn.benchmark = True

def main(gpu: int, num_workers: int, num_topics: int, from_scratch: bool, learning_rate: float,
         epochs: int, batch_size: int, verbose: bool, freq_id: int) -> None:
    """main method"""
    start = time.perf_counter()
    if verbose:
        # logging for gensim output
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    # set devices properly
    if gpu == 0:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if gpu == 1:
        device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

    # set model paths
    lda_path = "./models/freq_lda_model" if freq_id else "./models/lda_model"
    data_path = "./data/wiki_data_freq.tar" if freq_id else "./data/wiki_data.tar"
    dnn_path = "./models/dnn_model_freq" if freq_id else "./models/dnn_model"

    # print a summary of the chosen arguments
    print("\n\n\n"+"#"*50)
    print("## " + str(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p")))
    print("## System: {} CPU cores with {} GPUs on {}".format(torch.get_num_threads(),
                                                              torch.cuda.device_count(),
                                                              socket.gethostname()
                                                              ))
    if device == 'cpu':
        print("## Using: CPU with ID {}".format(device))
    else:
        print("## Using: {} with ID {}".format(torch.cuda.get_device_name(device=device), device))
    print("## Using {} workers for LDA computation".format(num_workers))
    print("## Num_topics: {}".format(num_topics))
    print("## Learning_rate: {}".format(learning_rate))
    print("## Batch_size: {}".format(batch_size))
    print("## Epochs: {}".format(epochs))
    if bool(freq_id):
        print("## Word ID: {}".format(freq_id))
    print("#"*50)
    print("\n\n\n")


    if not os.path.isfile(lda_path):
        # obtain a preprocessed list of words
        train_lda(num_workers, num_topics, freq_id)
    elif from_scratch:
        # obtain a preprocessed list of words
        train_lda(num_workers, num_topics, freq_id)
    else:
        print("[ a trained LDA model already exists. Train again? [y/n] ]")
        if from_scratch or input() == "y":
            # obtain a preprocessed list of words
            train_lda(num_workers, num_topics, freq_id)


    if not os.path.isfile(data_path):
        # save the lda model data as training data with labels
        save_train_data(freq_id=freq_id)
    elif from_scratch:
        # save the lda model data as training data with labels
        save_train_data(freq_id=freq_id)
    else:
        print("[ training data/labels already exists. Save them again? [y/n] ]")
        if from_scratch or input() == "y":
            # save the lda model data as training data with labels
            save_train_data(freq_id=freq_id)


    if not os.path.isfile(dnn_path):
        # train the DNN model on the lda dataset
        train(epochs=epochs,
              learning_rate=learning_rate,
              batch_size=batch_size,
              num_topics=num_topics,
              device_name=device,
              model_path=dnn_path,
              freq_id=freq_id)
    elif from_scratch:
        # train the DNN model on the lda dataset
        train(epochs=epochs,
              learning_rate=learning_rate,
              batch_size=batch_size,
              num_topics=num_topics,
              device_name=device,
              model_path=dnn_path,
              freq_id=freq_id)
    else:
        print("[ a trained DNN model already exists. Train again? [y/n] ]")
        if from_scratch or input() == "y":
            # train the DNN model on the lda dataset
            train(epochs=epochs,
                  learning_rate=learning_rate,
                  batch_size=batch_size,
                  num_topics=num_topics,
                  device_name=device,
                  model_path=dnn_path,
                  freq_id=freq_id)

    # evaluate both the lda and the dnn model and print their top topics
    evaluate(num_topics, is_freq=bool(freq_id))

    end = time.perf_counter()
    duration = (np.round(end - start) / 60.) / 60.
    print(f"\nComputation time: {duration:0.4f} hours")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", "-g", help="GPU", type=int, default=0)
    parser.add_argument("--freq_id", "-f", help="id of the word of which the frequency \
                        gets changed", type=int, default=None)
    parser.add_argument("--batch_size", "-b", help="batch size", type=int, default=512)
    parser.add_argument("--epochs", "-e", help="training epochs", type=int, default=100)
    parser.add_argument("--learning_rate", "-l", help="learning rate", type=float, default=0.01)
    parser.add_argument("--num_workers", "-w", help="number of workers for lda",
                        type=int, default=4)
    parser.add_argument("--num_topics", "-t", help="number of topics for lda",
                        type=int, default=100)
    parser.add_argument("--from_scratch", "-s", help="train lda from scratch",
                        action='store_true', default=False)
    parser.add_argument("--verbose", "-v", help="set gensim to verbose mode",
                        action='store_true', default=False)


    args = parser.parse_args()
    main(**vars(args))
