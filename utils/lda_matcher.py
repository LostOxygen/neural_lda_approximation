"""library module for LdaMatcher class and functionalities"""
# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from gensim.corpora import Dictionary


class LdaMatcher:
    """Class to hold and perform matching functionality for LDA's

    Attributes:

    """
    def __init__(self, lda_list: list, threshold: float, num_topics: int):
        """Inits the LdaMatcher with a list of N LDA's and the threshold
           used for the matching (to decide if a topic is similar enough
           to another).
        """
        self.lda_list = lda_list
        self.threshold = threshold
        self.num_topics = num_topics
        self.dictionary = Dictionary.load_from_text("./data/wikipedia_dump/wiki_wordids.txt")
        self.mapping = self.__create_mapping()


    def __create_mapping(self) -> torch.Tensor:
        """private method to create a mapping between every topic of all LDA's"""
        print("INFO: Create mapping for LdaMatcher")

         # preallocate a NUM_LDA x NUM_LDA x NUM_TOPICS x NUM_TOPICS tensor to save the mapping
        mapping = torch.zeros(
            (len(self.lda_list),
             len(self.lda_list),
             self.num_topics)
        )

        # mapping from the first LDA
        for idx, lda1 in enumerate(tqdm(self.lda_list, desc="First LDA", leave=False)):
            # to the second LDA
            for idy, lda2 in enumerate(tqdm(self.lda_list, desc="Second LDA", leave=False)):
                for topic_id in tqdm(range(self.num_topics), desc="Topics", leave=False):
                    # if the two LDA's are the same, skip iteration and use the identity mapping
                    if lda1 == lda2:
                        # identity mapping
                        mapping[idx, idy, topic_id] = topic_id
                        continue

                    else:
                        # and if the LDA's are not the same, compare it to every other word-topic
                        # distribution of the second LDA and find the best (if its < threshold)
                        difference = float("inf")
                        # initialize the first word-topic distribution
                        topics1 = lda1.get_topic_terms(topicid=topic_id, topn=len(self.dictionary))
                        word_prob_vec1 = torch.zeros(len(self.dictionary)) + 10e-16
                        for word_tuple1 in topics1:
                            # assign the word probabilites to their ID in the vector
                            word_prob_vec1[word_tuple1[0]] = torch.FloatTensor([word_tuple1[1]])

                        # the topic itself. Gets updated if a better is found
                        best_topic = topic_id
                        # iterate over the topics for the second LDA
                        for comp_topic_id in range(self.num_topics):
                            # initialize the start values and differences
                            topics2 = lda2.get_topic_terms(topicid=comp_topic_id,
                                                           topn=len(self.dictionary))
                            word_prob_vec2 = torch.zeros(len(self.dictionary)) + 10e-16
                            # iterate over every topic of the second LDA to calculate
                            # their difference and find the best
                            for word_tuple2 in topics2:
                                # assign the word probabilites to their ID in the vector
                                word_prob_vec2[word_tuple2[0]] = torch.FloatTensor([word_tuple2[1]])

                            curr_difference = self.__kl_div(word_prob_vec1, word_prob_vec2)

                            if curr_difference < difference and curr_difference <= self.threshold:
                                # if the current diff. is better, update the reference diff.
                                # and the best topic id
                                difference = curr_difference
                                best_topic = comp_topic_id

                        mapping[idx, idy, topic_id] = best_topic
        return mapping


    def get_mapping(self) -> torch.Tensor:
        """helper method which returns the generated LDA mapping"""
        return self.mapping


    def __kl_div(self, y: float, y_hat: float) -> float:
        """standard kullback leibler divergence loss as described in:
           https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html
        """
        return F.kl_div(y.log(), y_hat, None, None, reduction="sum")
