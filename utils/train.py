"""library module for training a given dnn model"""
import os
import json
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pkbar
import gensim
import webdataset as wds
from utils.network import DNN, CustomCrossEntropy
from utils.words import TEST_SET, DATA_PATH, LABEL_PATH
from utils.dataset import TensorDataset

SAVE_EPOCHS = [0, 25, 50, 75, 100, 125, 150, 175, 200]


def save_model(net: nn.Sequential) -> None:
    """
    helper function which saves the given net in the specified path.
    if the path does not exists, it will be created.
    :param path: path where the model should be saved. Gets created if non-existent
    :param file_name: name as which the model should be saved
    :param net: object of the model
    :return: None
    """
    print("[ Saving Model ]")
    state = {
        'net': net.state_dict()
    }
    if not os.path.isdir('./models'):
        os.mkdir('./model_saves')
    torch.save(state, "./models/{}".format("dnn_model"))


def get_model(num_topics: int, input_dim: int) -> nn.Sequential:
    """
    helper function to create and initialize the model.
    :return: returns the loaded model
    """
    net = DNN(num_topics, input_dim)
    net = net.to(device)
    return net


def adjust_learning_rate(optimizer, epoch: int, epochs: int, learning_rate: int) -> None:
    """
    helper function to adjust the learning rate
    according to the current epoch to prevent overfitting.
    :paramo ptimizer: object of the used optimizer
    :param epoch: the current epoch
    :param epochs: the total epochs of the training
    :param learning_rate: the specified learning rate for the training

    :return: None
    """
    new_lr = learning_rate
    if epoch >= np.floor(epochs*0.5):
        new_lr /= 10
    if epoch >= np.floor(epochs*0.75):
        new_lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def get_loaders(batch_size: int, dictionary: dict) -> DataLoader:
    """
    helper function to create dataset loaders
    :param batch_size: batch size which should be used for the dataloader
    :return: dataloader with the specified dataset
    """
    train_data_path = "./data/wiki_data.tar"

    test_data = []
    for i in TEST_SET:
        with open(os.path.join(DATA_PATH, i)) as file:
            dump = json.load(file)
        empty = np.zeros(len(dictionary))
        for key, val in dump.items():
            empty[int(key)] = float(val)
        test_data.append(empty)
    test_data = torch.FloatTensor(np.array(test_data)).to(device)

    test_labels = []
    for i in TEST_SET:
        with open(os.path.join(LABEL_PATH, i)) as file:
            # read the line and sanitize the string and convert it back to an int list
            tmp_str = file.readlines()
            tmp_str = list(map(float, tmp_str[0].replace("[", "").replace("]", "").split(",")))
            test_labels.append(tmp_str)
    test_labels = torch.FloatTensor(test_labels).to(device)

    train_dataset = wds.WebDataset(train_data_path).shuffle(1000).decode().to_tuple("input.pyd",
                                                                                    "output.pyd")
    test_dataset = TensorDataset(test_data, test_labels)

    train_loader = DataLoader((train_dataset.batched(batch_size)), batch_size=None, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=0)
    return train_loader, test_loader


def train(epochs: int, learning_rate: int, batch_size: int, num_topics: int,
          device_name: str) -> None:
    """
    Main method to train the model with the specified parameters. Saves the model in every
    epoch specified in SAVE_EPOCHS. Prints the model status during the training.
    :param epochs: specifies how many epochs the training should last
    :param learning_rate: specifies the learning rate of the training
    :param batch_size: specifies the batch size of the training data (default = 128)
    :param num_topics: number of learned topics by the lda model
    :param device_name: sets the device on which the training should be performed

    :return: None
    """
    print("[ Initialize Training ]")
    # create a global variable for the used device
    global device
    device = device_name

    # initializes the model, loss function, optimizer and dataset
    ldamodel = gensim.models.LdaMulticore.load('./models/lda_model')
    dictionary = ldamodel.id2word
    # input dimension of the model is the length of the dictionary
    net = get_model(num_topics, len(dictionary))

    criterion = CustomCrossEntropy()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    train_loader, test_loader = get_loaders(batch_size, dictionary)

    for epoch in range(0, epochs):
        # every epoch a new progressbar is created
        # also, depending on the epoch the learning rate gets adjusted before
        # the network is set into training mode
        kbar = pkbar.Kbar(target=int(5132412/batch_size), epoch=epoch, num_epochs=epochs,
                          width=20, always_stateful=True)
        adjust_learning_rate(optimizer, epoch, epochs, learning_rate)
        net.train()
        correct = 0
        total = 0
        running_loss = 0.0

        # iterates over a batch of training data
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # convert the sparse input vector back into a dense format
            inputs = inputs.to_dense()
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # calculate the current running loss as well as the total accuracy
            # and update the progressbar accordingly
            running_loss += loss.item()
            total += targets.size(0)
            correct += torch.isclose(F.softmax(outputs, dim=-1), targets,
                                     rtol=1e-04, atol=1e-05).sum().item()

            kbar.update(batch_idx, values=[("loss", running_loss/(batch_idx+1)),
                                           ("acc", 100. * correct/total)])
        # calculate the test accuracy of the network at the end of each epoch
        with torch.no_grad():
            net.eval()
            t_total = 0
            t_correct = 0
            for _, (inputs_t, targets_t) in enumerate(test_loader):
                inputs_t, targets_t = inputs_t.to(device), targets_t.to(device)
                outputs_t = net(inputs_t)

                t_total += targets_t.size(0)
                t_correct += torch.isclose(F.softmax(outputs_t, dim=-1), targets_t,
                                           rtol=1e-04, atol=1e-05).sum().item()
            print("-> test acc: {}".format(100.*t_correct/t_total))

        # save the model at the end of the specified epochs as well as at
        # the end of the whole training
        if epoch in SAVE_EPOCHS:
            save_model(net)
    save_model(net)

    # calculate the test accuracy of the network at the end of the training
    with torch.no_grad():
        net.eval()
        t_total = 0
        t_correct = 0
        for _, (inputs_t, targets_t) in enumerate(test_loader):
            inputs_t, targets_t = inputs_t.to(device), targets_t.to(device)
            outputs_t = net(inputs_t)

            t_total += targets_t.size(0)
            t_correct += torch.isclose(F.softmax(outputs_t, dim=-1), targets_t,
                                       rtol=1e-04, atol=1e-05).sum().item()

    print("Final accuracy: Train: {} | Test: {}".format(100.*correct/total, 100.*t_correct/t_total))
