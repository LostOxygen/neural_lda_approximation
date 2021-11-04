"""main file to run the lda approximation"""
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import time
import os
import datetime
import argparse
import torch
import numpy as np
from utils.words import get_word_list

def main(gpu: int, workers: int, num_topics: int) -> None:
    """main method"""
    words_list = get_word_list()
    print(words_list)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", "-g", help="GPU", type=int, default=0)
    parser.add_argument("--workers", "-w", help="number of workers for lda",
                        type=int, default=4)
    parser.add_argument("--num_topics", "-t", help="number of topics for lda",
                        type=int, default=20)

    args = parser.parse_args()
    main(**vars(args))
