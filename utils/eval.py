"""helper module to evaluate the lda and dnn model"""
from typing import Tuple
from pprint import pprint
import numpy as np
import gensim
from gensim.models import LdaMulticore
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.words import get_word_list
from utils.network import DNN

def get_models(num_topics: int) -> Tuple[LdaMulticore, nn.Sequential]:
    """helper function to load and return the previous trained LDA and DNN models
       :param num_topics:
       :return: a tuple of both the LDA and the DNN model
    """
    lda_model = gensim.models.LdaMulticore.load('./models/lda_model')

    dnn_model = DNN(num_topics, len(lda_model.id2word))
    checkpoint = torch.load('./models/dnn_model', map_location=lambda storage, loc: storage)
    dnn_model.load_state_dict(checkpoint['net'], strict=False)
    dnn_model.eval()

    return lda_model, dnn_model

def evaluate(num_topics: int) -> None:
    """helper function to evaluate the lda and dnn model and calculate the top
       topics for a given test text.
       :param num_topics: number of topics which the lda model tries to match
       :return: None
    """
    lda_model, dnn_model = get_models(num_topics)

    test_doc = get_word_list(is_train=False)
    dictionary = lda_model.id2word
    bow_list = list(map(lambda x: dictionary.doc2bow(x), test_doc))

    doc_topics_lda = lda_model.get_document_topics(bow_list)
    top_lda_topics = []
    print("\ntopic prediction of the lda model: ")
    for topic in doc_topics_lda[0]:
        top_lda_topics.append(topic)

    top_lda_topics = sorted(top_lda_topics, key=lambda x: x[1], reverse=True)
    pprint(top_lda_topics)

    eval_data = []
    for bow_elem in bow_list:
        empty = np.zeros(len(dictionary))
        for key, val in bow_elem.items():
            empty[int(key)] = float(val)
        eval_data.append(empty)
    eval_data = torch.FloatTensor(eval_data)

    doc_topics_dnn = F.softmax(dnn_model(torch.Tensor(eval_data)).detach()[0], dim=-1)
    print("\ntopic prediction of the dnn model: ")
    topk_topics = doc_topics_dnn.topk(len(doc_topics_lda[0]))
    for i in range(len(doc_topics_lda[0])):
        print((topk_topics[1][i].item(), topk_topics[0][i].item()))
