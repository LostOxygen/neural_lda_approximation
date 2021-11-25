"""helper module to evaluate the lda and dnn model"""
from typing import Tuple
import gensim
from gensim.models import LdaMulticore
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import webdataset as wds
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
    test_data_path = "./data/wiki_test.tar"
    test_dataset = wds.WebDataset(test_data_path).decode().shuffle(1000).to_tuple("input.pyd",
                                                                                  "output.pyd")
    test_loader = DataLoader((test_dataset.batched(1)), batch_size=None, num_workers=0)
    _, test_bow = next(enumerate(test_loader))
    # convert sparse tensor back into dense form
    test_bow = test_bow[0].to_dense()
    # convert tensor back into bag of words list for the lda model
    test_bow_lda = test_bow[0].tolist()
    test_bow_lda = [(id, int(counting)) for id, counting in enumerate(test_bow_lda)]

    doc_topics_lda = lda_model.get_document_topics(list(test_bow_lda))
    top_lda_topics = []
    print("\ntopic prediction of the lda model: ")
    for topic in doc_topics_lda:
        top_lda_topics.append(topic)

    sorted_lda_topics = sorted(top_lda_topics, key=lambda x: x[1], reverse=True)
    for topic in sorted_lda_topics:
        print((lda_model.id2word[topic[0]], topic[1]))

    doc_topics_dnn = F.softmax(dnn_model(torch.Tensor(test_bow)).detach()[0], dim=-1)
    print("\ntopic prediction of the dnn model: ")
    topk_topics = doc_topics_dnn.topk(len(doc_topics_dnn))
    for i in range(len(top_lda_topics)):
        print((lda_model.id2word[topk_topics[1][i].item()], topk_topics[0][i].item()))
