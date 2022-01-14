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
from utils.network import DNN, CustomCrossEntropy, KLDivLoss
from torch.nn  import BCEWithLogitsLoss, BCELoss

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


def get_norm_batch(batch, norm):
    """function from Advertorch for normalize batches"""
    batch_size = batch.size(0)
    return batch.abs().pow(norm).view(batch_size, -1).sum(dim=1).pow(1. / norm)


def normalize_probs(probs):
    """helper function to normalize probability vectors"""
    factor = 1 / torch.sum(probs)
    return probs * factor


def attack(model: nn.Sequential, bow: torch.FloatTensor, device: str, attack_id: int,
           advs_eps: float, num_topics: int, lda_model: LdaMulticore, l2_attack: bool,
           max_iteration: int, prob_attack: bool) -> torch.FloatTensor:
    """helper function to create adversarial examples to attack the DNN and LDA model
    :param model: the trained dnn model from which we use the gradient for the attack
    :param attack_id: sets the target word id for the adversarial attack
    :param advs_eps: epsilon value for the adverarial attack
    :param num_topics: number of topics which the lda model tries to match
    :param lda_model: the lda model which gets proxied by the dnn
    :param l2_attack: flag to specifiy if the l2 attack or the linf should be used
    :param prob_attack: flag which activates a whole prob. dist. as a target instead of a single

    :return: Tensor with the manipulated word frequencies
    """
    print("\n######## [ generating adversarial example for topic {} ] ########\n".format(attack_id))
    step_size = 100 # attack step size
    ce_boundary = 0.05 # cross entropy boundary
    epsilon = float(advs_eps)
    bow = bow.to(device)
    model = copy.deepcopy(model).to(device)

    if prob_attack:
        prob_dist = torch.zeros(num_topics)
        prob_dist_base = torch.zeros(num_topics)

        # obtain the normal topic distribution of the LDA model
        bow_lda_list = [(id, int(counting)) for id, counting in enumerate(bow[0].tolist())]
        topic_dist = lda_model.get_document_topics(bow_lda_list)

        # small noise constant is getting added onto the non-zero topic probabilities
        noise = torch.abs(torch.randn(len(topic_dist))) * 1e-3

        # write values back in the whole probability distribution vector add the noise and normalize
        for idx, topic_tuple in enumerate(topic_dist):
            prob_dist_base[topic_tuple[0]] = torch.tensor(topic_tuple[1])
            prob_dist[topic_tuple[0]] = torch.tensor(topic_tuple[1]) + noise[idx]

        prob_dist_base = prob_dist_base.to(device)
        target = normalize_probs(prob_dist).to(device)
        loss_class = BCELoss()

        print("Target Distribution:")
        print([target.sort(descending=True)])
        reference_cross_entropy = loss_class(prob_dist_base, target)
        print("Cross-Entropy betw. Base and Target: {}".format(reference_cross_entropy))

    else:
        target = torch.LongTensor([attack_id]).to(device)
        loss_class = nn.CrossEntropyLoss(reduction="sum")


    if l2_attack:
        current_iteration = 0
        cross_entropy_value = 0.0
        delta = torch.zeros_like(bow)
        delta.requires_grad_()

        while True:
            current_iteration += 1
            current_nonzeros = torch.count_nonzero(torch.round((bow+delta)[0].detach()))
            print("-> current attack iteration: {} with " \
                  "current nonzero elem.: {} " \
                  "and CE: {} ".format(current_iteration, current_nonzeros,
                                       cross_entropy_value), end="\r")
            if prob_attack:
                outputs = F.softmax(model(bow + delta), dim=-1)
                loss = -loss_class(outputs.squeeze().cpu(), target.cpu())
            else:
                outputs = model(bow + delta)
                loss = -loss_class(outputs, target)
            loss.backward()

            # perform the attack
            grad = delta.grad.data
            # normalize the gradient on L2 norm
            norm = get_norm_batch(batch=grad, norm=2)
            norm = torch.max(norm, torch.ones_like(norm) * 1e-6)
            grad = torch.multiply(1. / norm, grad)

            # perform a step
            delta.data = delta.data + torch.multiply(grad, step_size)
            delta.data = torch.clamp(bow.data + delta.data, 0.0, 100000.0) - bow.data

            # project back into epsilon constraint
            if epsilon is not None:
                delta_norm = get_norm_batch(batch=grad, norm=2)
                factor = torch.min(epsilon / delta_norm, torch.ones_like(delta_norm))
                delta.data = torch.multiply(delta.data, factor)

            # check if the attack was successful on the original lda
            rounded_advs = torch.round(bow + delta)
            test_bow_lda = rounded_advs[0].tolist()
            test_bow_lda = [(id, int(counting)) for id, counting in enumerate(test_bow_lda)]
            topics_lda = lda_model.get_document_topics(list(test_bow_lda))

            if prob_attack:
                # calculate the cross entropy value between the adversarial example LDA output
                # and the target
                prob_dist_lda = torch.zeros(num_topics).to(device)
                for idx, topic_tuple in enumerate(topics_lda):
                    prob_dist_lda[topic_tuple[0]] = torch.tensor(topic_tuple[1])
                cross_entropy_value = loss_class(prob_dist_lda, target)
                if cross_entropy_value < reference_cross_entropy:
                    advs = rounded_advs.detach()
                    print("\n-> attack converged in {} iterations!".format(current_iteration))
                    print("Cross-Entropy betw. Poison and Target: {}".format(cross_entropy_value))
                    return advs.cpu(), True
            else:
                sorted_lda_topics = sorted(topics_lda, key=lambda x: x[1], reverse=True)
                if sorted_lda_topics[0][0] == target:
                    advs = rounded_advs.detach()
                    print("\n-> attack converged in {} iterations!".format(current_iteration))
                    return advs.cpu(), True

            if current_iteration >= max_iteration:
                advs = rounded_advs.detach()
                print("\n-> max iterations reached!")
                return advs.cpu(), False

    else:
        delta = torch.zeros_like(bow)
        delta.requires_grad_()
        current_iteration = 0

        while True:
            current_iteration += 1
            print("-> current attack iteration: {} with " \
                  "current nonzero elem.: {}".format(current_iteration,
                                                     torch.count_nonzero((bow+delta)[0].detach()),
                                                     end="\r"))
            outputs = model(bow + delta)
            loss = -loss_class(outputs, target)
            loss.backward()

            grad_sign = delta.grad.data.sign()
            # delta.data = delta.data + batch_multiply(1.0, grad_sign)
            delta.data = delta.data + grad_sign
            delta.data = torch.clamp(delta.data, torch.tensor(epsilon).to(device))
            delta.data = torch.clamp(bow.data + delta.data, 0.0, 1000000.0) - bow.data
            delta.grad.data.zero_()

            # check if the attack was successful on the original lda
            rounded_advs = torch.round(bow+delta)
            test_bow_lda = rounded_advs[0].tolist()
            test_bow_lda = [(id, int(counting)) for id, counting in enumerate(test_bow_lda)]
            topics_lda = lda_model.get_document_topics(list(test_bow_lda))
            sorted_lda_topics = sorted(topics_lda, key=lambda x: x[1], reverse=True)
            if prob_attack:
                raise NotImplementedError("Attack isn't implemented for using the whole prob." \
                                          "Please use --l2_attack flag.")

            if sorted_lda_topics[0][0] == target:
                advs = (bow+delta).detach()
                print("\n-> attack converged in {} iterations!".format(current_iteration))
                return advs.cpu(), True

            if current_iteration >= max_iteration:
                advs = (bow+delta).detach()
                print("\n-> max iterations reached!")
                return advs.cpu(), False


def evaluate(num_topics: int, attack_id: int, random_test: bool, advs_eps: float,
             device: str, l2_attack: bool, max_iteration: int, prob_attack: bool) -> None:
    """helper function to evaluate the lda and dnn model and calculate the top
       topics for a given test text.
       :param num_topics: number of topics which the lda model tries to match
       :param attack_id: sets the target word id for the adversarial attack
       :param random_test: flag enables random test documents
       :param device: the device on which the computation happens
       :param advs_eps: epsilon value for the adverarial attack
       :param l2_attack: flag to specifiy if the l2 attack or the linf should be used
       :param prob_attack: flag which activates a whole prob. dist. as a target instead of a single

       :return: None
    """
    lda_model, dnn_model = get_models(num_topics, None)
    # yes the train set is used on purpose!
    test_data_path = "./data/wiki_data.tar"

    if random_test or bool(attack_id) or l2_attack:
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
    if bool(attack_id) or prob_attack:
        manipulated_bow, success_flag = attack(dnn_model, test_bow, device, attack_id, advs_eps,
                                               num_topics, lda_model, l2_attack, max_iteration,
                                               prob_attack)
        if not success_flag:
            return False

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

        return True
