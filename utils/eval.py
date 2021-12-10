"""helper module to evaluate the lda and dnn model"""
from typing import Tuple
import copy
import gensim
from gensim.models import LdaMulticore
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import webdataset as wds
from advertorch.attacks import L2PGDAttack
from utils.network import DNN, CustomCrossEntropy

def get_models(num_topics: int, is_freq: bool) -> Tuple[LdaMulticore, nn.Sequential]:
    """helper function to load and return the previous trained LDA and DNN models
       :param num_topics: number of topics which the lda model tries to match
       :param is_freq: if flag is set the model with the changed frequency should gets loaded
       :return: a tuple of both the LDA and the DNN model
    """
    lda_model = gensim.models.LdaMulticore.load('./models/lda_model')

    dnn_model = DNN(num_topics, len(lda_model.id2word))
    if is_freq:
        checkpoint = torch.load('./models/dnn_model_freq',
                                map_location=lambda storage, loc: storage)
    else:
        checkpoint = torch.load('./models/dnn_model', map_location=lambda storage, loc: storage)
    dnn_model.load_state_dict(checkpoint['net'], strict=False)
    dnn_model.eval()

    return lda_model, dnn_model

def attack(model: nn.Sequential, bow: torch.FloatTensor, device: str,
           attack_id: int, advs_eps: float, advs_iters: int) -> torch.FloatTensor:
    """helper function to create adversarial examples to attack the DNN and LDA model
    :param model: the trained dnn model from which we use the gradient for the attack
    :param attack_id: sets the target word id for the adversarial attack
    :param advs_eps: epsilon value for the adverarial attack
    :param advs_iters: iterations of pgd

    :return: Tensor with the manipulated word frequencies
    """
    print("\n########## [ generating adversarial example.. ] ##########")
    loss_class = CustomCrossEntropy()
    iters = advs_iters
    epsilon = advs_eps
    bow = bow.to(device)
    target = torch.LongTensor([attack_id]).to(device)
    model = copy.deepcopy(model).to(device)
    # initialize the adversary class
    adversary = L2PGDAttack(model, loss_fn=loss_class, eps=epsilon, nb_iter=iters,
                            eps_iter=(epsilon/10.), rand_init=True,
                            clip_min=0.0, clip_max=1000.0, targeted=True)
    advs = adversary.perturb(bow, target).detach().cpu()
    # advs = torch.round(advs)

    return advs


def evaluate(num_topics: int, attack_id: int, random_test: bool,
             advs_eps: float, advs_iters: int, device: str) -> None:
    """helper function to evaluate the lda and dnn model and calculate the top
       topics for a given test text.
       :param num_topics: number of topics which the lda model tries to match
       :param attack_id: sets the target word id for the adversarial attack
       :param random_test: flag enables random test documents
       :param device: the device on which the computation happens
       :param advs_eps: epsilon value for the adverarial attack

       :param advs_iters: iterations of pgd
       :return: None
    """
    lda_model, dnn_model = get_models(num_topics, None)
    test_data_path = "./data/wiki_test.tar"

    if random_test:
        test_dataset = wds.WebDataset(test_data_path).decode().shuffle(1000).to_tuple("input.pyd",
                                                                                      "output.pyd")
    else:
        test_dataset = wds.WebDataset(test_data_path).decode().to_tuple("input.pyd", "output.pyd")
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
    prob_list_lda = []
    for topic in sorted_lda_topics:
        print(("id: {}".format(topic[0]),
               lda_model.id2word[topic[0]],
               "prob: {}".format(topic[1])
               ))
        prob_list_lda.append(topic[1])

    doc_topics_dnn = F.softmax(dnn_model(torch.Tensor(test_bow)).detach()[0], dim=-1)
    print("\ntopic prediction of the dnn model: ")
    topk_topics = doc_topics_dnn.topk(len(doc_topics_dnn))
    prob_list_dnn = []
    for i in range(len(top_lda_topics)):
        print(("id: {}".format(topk_topics[1][i].item()),
               lda_model.id2word[topk_topics[1][i].item()],
               "prob: {}".format(topk_topics[0][i].item())
               ))

    if bool(attack_id):
        manipulated_bow = attack(dnn_model, test_bow, device,
                                 attack_id, advs_eps, advs_iters)

        # convert tensor back into bag of words list for the lda model
        test_bow_lda = manipulated_bow[0].tolist()
        test_bow_lda = [(id, int(counting)) for id, counting in enumerate(test_bow_lda)]

        doc_topics_lda = lda_model.get_document_topics(list(test_bow_lda))
        top_lda_topics = []
        print("\ntopic prediction of the lda model: ")
        for topic in doc_topics_lda:
            top_lda_topics.append(topic)

        sorted_lda_topics = sorted(top_lda_topics, key=lambda x: x[1], reverse=True)
        for topic in sorted_lda_topics:
            print(("id: {}".format(topic[0]),
                   lda_model.id2word[topic[0]],
                   "prob: {}".format(topic[1])
                   ))

        doc_topics_dnn = F.softmax(dnn_model(torch.Tensor(manipulated_bow)).detach()[0], dim=-1)
        print("\ntopic prediction of the dnn model on adversarial example: ")
        topk_topics = doc_topics_dnn.topk(len(doc_topics_dnn))
        for i in range(len(top_lda_topics)):
            print(("id: {}".format(topk_topics[1][i].item()),
                   lda_model.id2word[topk_topics[1][i].item()],
                   "prob: {}".format(topk_topics[0][i].item())
                   ))
