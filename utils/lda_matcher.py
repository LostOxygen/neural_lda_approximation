"""library module for LdaMatcher class and functionalities"""
# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from gensim.corpora import Dictionary
from gensim.models import LdaMulticore


class LdaMatcher:
    """Class to hold and perform matching functionality for LDA's

    Attributes:
        lda_list: list of LDAs to create the matching for
        threshold: if the similarity between two lda topics
        num_topics: number of LDA topics
        core_topic_candidates: a list of core_topic candidates, gets updates during the mapping
        dictionary: the gensim dictionary of the LDAs
        mapping: final tensor with topic mapping for every LDA combination
        core_topics: final dictionary with the core topics and their word distributions
    """
    def __init__(self, lda_list: list, threshold: float, num_topics: int, dictionary: Dictionary):
        """Inits the LdaMatcher with a list of N LDA's and the threshold
           used for the matching (to decide if a topic is similar enough
           to another).
        """
        self.lda_list = lda_list
        self.threshold = threshold
        self.num_topics = num_topics
        self.core_topic_candidates = list(range(self.num_topics)) # used to find the core topics
        self.dictionary = dictionary
        self.mapping = self.__create_mapping()
        self.core_topics = self.__find_core_topics()


    def __kl_div(self, y: float, y_hat: float) -> torch.FloatTensor:
        """standard kullback leibler divergence loss as described in:
           https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html
        """
        return F.kl_div(y.log(), y_hat, None, None, reduction="sum")


    def __find_core_topics(self) -> dict:
        """private method to loop over the generated mapping and list the topics which exist
           in every LDA.
        """
        print("INFO: Listing the core topics of all LDAs")

        # core_topics is a dictionary where the topic ID is the key and a list of tuples are
        # the value. Every tuple contains the word ID with the corresponding probability
        core_topics = dict()

        for lda in tqdm(self.lda_list, desc="LDAs", leave=False):
            for topic_id in self.core_topic_candidates:
                lda_topic_terms = lda.get_topic_terms(topicid=topic_id, topn=len(self.dictionary))
                # if the entry does not exist, add it
                if core_topics.get(topic_id) is None:
                    core_topics[topic_id] = lda_topic_terms
                # if the entry already exists, add the probabilites to the accord word IDs
                else:
                    core_topics[topic_id] = [(tuple[0], tuple[1] + new_probs[1]) for \
                                             tuple, new_probs in zip(core_topics[topic_id],
                                                                     lda_topic_terms)]

        # normalize the core_topics for every topic word distributen w.r.t
        # to the length of the candidate list
        for topic_id in self.core_topic_candidates:
            # build a vector with all probs for a specific topic
            topic_probs = [topic_tuple[1] for topic_tuple in core_topics[topic_id]]
            # then use the softmax to normalize the probability distribution
            topic_probs = F.softmax(torch.tensor(topic_probs), dim=-1)

            core_topics[topic_id] = [(tuple[0], tuple[1]) for tuple in \
                                     zip(core_topics[topic_id], topic_probs)]

        return core_topics


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
                        # if the topic got mapped, it is no longer a candidate for the core topics
                        # therefore it's topic ID gets removed from the list
                        if topic_id != best_topic and topic_id in self.core_topic_candidates:
                            self.core_topic_candidates.remove(topic_id)
        return mapping


    def __get_ranking(self, topics_i: list, topics_j: list):
        """private helper method which calculates and returns the ranking based on the
           number of intersections between the two topic lists
        """
        topics_i = torch.tensor(topics_i)
        topics_j = torch.tensor(topics_j)

        return torch.dot(topics_i, topics_j.T)


    def get_similar_document(self, article_i: list, article_j: list) -> tuple:
        """helper method to measure the similarity of two documents by using their
           topic intersection length.
        """
        lda = self.lda_list[0]
        # obtain the topic lists of the documents and save them in a list
        # also only keep their topic ID and cut of the probabilities
        # (topic embedding of the articles)
        topics_list = [topic_tuple[0] for topic_tuple in \
                       [lda.get_document_topics(article, minimum_probability=0.005) \
                        for article in article_i]]

        topics_targ = [topic_tuple[0] for topic_tuple in \
                       [lda.get_document_topics(article, minimum_probability=0.005) \
                        for article in article_j]]

        # iterate over the topics of every document and compare their topics to the target
        document_ranking = list()
        for _, targ_topics in enumerate(topics_targ):
            for doc_id, topics in enumerate(topics_list):
                document_ranking.append((doc_id, self.__get_ranking(topics, targ_topics)))

        # sort the document ids w.r.t their ranking values
        document_ranking.sort(key=lambda x: x[1])

        # return the best document with its ranking score
        return document_ranking[0]


    def get_mapping(self) -> torch.Tensor:
        """helper method which returns the generated LDA mapping"""
        return self.mapping


    def get_core_topic_ids(self) -> list:
        """helper method which returns the list with the plain core topic IDs"""
        return list(self.core_topics.keys())


    def get_core_topics(self) -> dict:
        """helper method which returns the core topic dictionary of every LDA"""
        return self.core_topics
