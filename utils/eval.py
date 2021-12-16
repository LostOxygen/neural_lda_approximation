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
# from advertorch.attacks import LinfPGDAttack, L2PGDAttack
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
           attack_id: int, advs_eps: float, advs_iters: int,
           lda_model: LdaMulticore) -> torch.FloatTensor:
    """helper function to create adversarial examples to attack the DNN and LDA model
    :param model: the trained dnn model from which we use the gradient for the attack
    :param attack_id: sets the target word id for the adversarial attack
    :param advs_eps: epsilon value for the adverarial attack
    :param advs_iters: iterations of pgd
    :param lda_model: the lda model which gets proxied by the dnn

    :return: Tensor with the manipulated word frequencies
    """
    print("\n########## [ generating adversarial example.. ] ##########")
    loss_class = nn.CrossEntropyLoss(reduction="sum")
    iters = advs_iters
    epsilon = advs_eps
    bow = bow.to(device)
    target = torch.LongTensor([attack_id]).to(device)
    model = copy.deepcopy(model).to(device)
    # initialize the adversary class
    # adversary = L2PGDAttack(model, loss_fn=loss_class, eps=epsilon, nb_iter=iters,
    #                         eps_iter=epsilon, rand_init=True,
    #                         clip_min=0.0, clip_max=1000.0, targeted=True)
    delta = torch.zeros_like(bow)
    delta.requires_grad_()
    current_iteration = 0

    #for ii in range(iters):
    while True:
        current_iteration += 1
        print("-> current attack iteration: {} with " \
              "current bow values: {}".format(current_iteration,
                                              (bow+delta)[0].detach()), end="\r")
        outputs = model(bow + delta)
        # check if the attack was successful on the original lda
        rounded_advs = torch.round(bow+delta)
        test_bow_lda = rounded_advs[0].tolist()
        test_bow_lda = [(id, int(counting)) for id, counting in enumerate(test_bow_lda)]
        topics_lda = lda_model.get_document_topics(list(test_bow_lda))
        sorted_lda_topics = sorted(topics_lda, key=lambda x: x[1], reverse=True)
        if sorted_lda_topics[0][0] == target:
            break

        # print(torch.argmax(F.softmax(outputs[0], dim=-1)))
        loss = -loss_class(outputs, target)
        loss.backward()

        grad_sign = delta.grad.data.sign()
        # delta.data = delta.data + batch_multiply(1.0, grad_sign)
        delta.data = delta.data + grad_sign
        delta.data = torch.clamp(delta.data, torch.tensor(epsilon).to(device))
        delta.data = torch.clamp(bow.data + delta.data, 0.0, 1000000.0) - bow.data
        delta.grad.data.zero_()

    advs = (bow+delta).detach().cpu()
    print("-> attack converged in {} iterations!".format(current_iteration))

    # advs = adversary.perturb(bow, target).detach().cpu()

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
    print("\ntopic prediction of the lda model: ")
    sorted_lda_topics = sorted(doc_topics_lda, key=lambda x: x[1], reverse=True)
    # collect the probabilities to calculate the cross entropy later on
    prob_tensor_lda = torch.zeros(num_topics)
    for topic in sorted_lda_topics:
        # and also print the ids, words and probs nicely
        print(("id: {}".format(topic[0]),
               lda_model.id2word[topic[0]],
               "prob: {}".format(topic[1])
               ))
        # insert the probabilities at their id position
        prob_tensor_lda[topic[0]] = torch.tensor(topic[1])
    print("total probability: {}".format(prob_tensor_lda.sum()))
    print("-> highest class: {} with prob: {}".format(sorted_lda_topics[0][0],
                                                      sorted_lda_topics[0][1]))


    doc_topics_dnn = dnn_model(torch.Tensor(test_bow)).detach()[0]
    print("\ntopic prediction of the dnn model: ")
    topk_topics = F.softmax(doc_topics_dnn, dim=-1).topk(num_topics)
    # collect the probabilities to calculate the cross entropy later on
    prob_tensor_dnn = torch.zeros(num_topics)
    for i in range(num_topics):
        # and also print the ids, words and probs nicely
        # but print only as much as the lda did
        if not i > len(sorted_lda_topics):
            print(("id: {}".format(topk_topics[1][i].item()),
                   lda_model.id2word[topk_topics[1][i].item()],
                   "prob: {}".format(topk_topics[0][i].item())
                   ))
        # insert the probabilities at their id position
        prob_tensor_dnn[topk_topics[1][i].item()] = topk_topics[0][i].item()
    print("total probability: {}".format(prob_tensor_dnn.sum()))
    print("-> highest class: {} with prob: {}".format(topk_topics[1][0], topk_topics[0][0]))


    prob_tensor_lda = prob_tensor_lda.unsqueeze(dim=0)
    prob_tensor_dnn = prob_tensor_dnn.unsqueeze(dim=0)
    # the dnn output is already softmaxed, so it just has to be log'ed
    ce_score = -torch.mean(torch.sum(prob_tensor_lda * torch.log(prob_tensor_dnn), -1))
    print("\n-> Cross-Entropy between LDA and DNN: {}".format(ce_score))

# ----------------- start of the adversarial attack stuff ----------------------------
    if bool(attack_id):
        manipulated_bow = attack(dnn_model, test_bow, device, attack_id,
                                 advs_eps, advs_iters, lda_model)

        # convert tensor back into bag of words list for the lda model
        test_bow_lda = manipulated_bow[0].tolist()
        test_bow_lda = [(id, int(counting)) for id, counting in enumerate(test_bow_lda)]

        doc_topics_lda = lda_model.get_document_topics(list(test_bow_lda))
        print("\ntopic prediction of the lda model on advs. example: ")


        sorted_lda_topics = sorted(doc_topics_lda, key=lambda x: x[1], reverse=True)
        # collect the probabilities to calculate the cross entropy later on
        prob_tensor_lda = torch.zeros(num_topics)
        for topic in sorted_lda_topics:
            # and also print the ids, words and probs nicely
            print(("id: {}".format(topic[0]),
                   lda_model.id2word[topic[0]],
                   "prob: {}".format(topic[1])
                   ))
            # insert the probabilities at their id position
            prob_tensor_lda[topic[0]] = torch.tensor(topic[1])
        print("total probability: {}".format(prob_tensor_lda.sum()))
        print("-> highest class: {} with prob: {}".format(sorted_lda_topics[0][0],
                                                          sorted_lda_topics[0][1]))


        doc_topics_dnn = dnn_model(torch.Tensor(manipulated_bow)).detach()[0]
        print("\ntopic prediction of the dnn model on advs. example: ")
        topk_topics = F.softmax(doc_topics_dnn, dim=-1).topk(num_topics)
        # collect the probabilities to calculate the cross entropy later on
        prob_tensor_dnn = torch.zeros(num_topics)
        for i in range(num_topics):
            # and also print the ids, words and probs nicely
            # but print only as much as the lda did
            if not i > len(sorted_lda_topics):
                print(("id: {}".format(topk_topics[1][i].item()),
                       lda_model.id2word[topk_topics[1][i].item()],
                       "prob: {}".format(topk_topics[0][i].item())
                       ))
            # insert the probabilities at their id position
            prob_tensor_dnn[topk_topics[1][i].item()] = topk_topics[0][i].item()
        print("total probability: {}".format(prob_tensor_dnn.sum()))
        print("-> highest class: {} with prob: {}".format(topk_topics[1][0], topk_topics[0][0]))


        prob_tensor_lda = prob_tensor_lda.unsqueeze(dim=0)
        prob_tensor_dnn = prob_tensor_dnn.unsqueeze(dim=0)
        # the dnn output is already softmaxed, so it just has to be log'ed
        ce_score = -torch.mean(torch.sum(prob_tensor_lda * torch.log(prob_tensor_dnn), -1))
        print("\n-> Cross-Entropy between LDA and DNN: {}".format(ce_score))
