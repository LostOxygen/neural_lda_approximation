"""main file to run the lda approximation"""
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import time
import socket
import datetime
import argparse
import torch
import numpy as np
from utils.words import get_word_list
from utils.lda import train_lda

torch.backends.cudnn.benchmark = True

def main(gpu: int, num_workers: int, num_topics: int) -> None:
    """main method"""
    start = time.perf_counter()
    # set device properly
    if gpu == 0:
        DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if gpu == 1:
        DEVICE = 'cuda:1' if torch.cuda.is_available() else 'cpu'

    # print a summary of the chosen arguments
    print("\n\n\n"+"#"*50)
    print("## " + str(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p")))
    print("## System: {} CPU cores with {} GPUs on {}".format(torch.get_num_threads(),
                                                              torch.cuda.device_count(),
                                                              socket.gethostname()
                                                              ))
    if DEVICE == 'cpu':
        print("## Using: CPU with ID {}".format(DEVICE))
    else:
        print("## Using: {} with ID {}".format(torch.cuda.get_device_name(device=DEVICE), DEVICE))
    print("## Using {} workers for LDA computation".format(num_workers))
    print("## Num_topics: {}".format(num_topics))
    print("#"*50)
    print("\n\n\n")

    words_list = get_word_list()
    train_lda(num_workers, num_topics, words_list)

    end = time.perf_counter()
    duration = (np.round(end - start) / 60.) / 60.
    print(f"Computation time: {duration:0.4f} hours")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", "-g", help="GPU", type=int, default=0)
    parser.add_argument("--num_workers", "-w", help="number of workers for lda",
                        type=int, default=4)
    parser.add_argument("--num_topics", "-t", help="number of topics for lda",
                        type=int, default=20)

    args = parser.parse_args()
    main(**vars(args))
