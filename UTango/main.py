import random
from time import sleep

import torch
from sklearn.metrics import accuracy_score
from torch import nn
from tqdm import tqdm
import model
import os.path as osp
import os
import copy
import numpy as np


def start_training(dataset):
    random.shuffle(dataset)
    train_dataset = dataset[:25] # Change based on your data split, if you want to have a validate set, you can also setup a validation set for it.
    test_dataset = dataset[25:] # Change based on your data split, if you want to have a validate set, you can also setup a validation set for it.
    model_ = model.UTango(h_size=128, max_context=5, drop_out_rate=0.5, gcn_layers=3)
    train(epochs=1, trainLoader=train_dataset, testLoader=test_dataset, model=model_, learning_rate=0.0001)


def evaluate_metrics(model, test_loader):
    model.eval()
    with torch.no_grad():
        acc = 0
        for data in tqdm(test_loader):
            out = model(data[:-1])
            temp_acc = 0
            for i in range(len(out)):
                loop_set = loop_calculation(out[i], data[-1][i])
                max_acc = -999
                for pos_ in loop_set:
                    tmp_acc = accuracy_score(pos_, data[-1][i])
                    if tmp_acc > max_acc:
                        max_acc = tmp_acc
                temp_acc += max_acc
            temp_acc = temp_acc/len(out)
            acc += temp_acc
        acc = acc/len(test_loader)
        sleep(0.1)
        print("Average Accuracy: ", acc)


def loop_calculation(input_1, input_2):
    out_ = []
    input_set = set(input_1)
    label_set = set(input_2)
    pairs = loop_check(label_set, input_set)
    for pair in pairs:
        tem_input = copy.deepcopy(input_1)
        changed = np.zeros(len(tem_input))
        for pair_info in pair:
            original_label = pair_info[0]
            replace_label = pair_info[1]
            for i in range(len(tem_input)):
                if tem_input[i] == original_label and changed[i] == 0:
                    tem_input[i] = replace_label
                    changed[i] = 1
        for i in range(len(changed)):
            if changed[i] == 0:
                tem_input[i] = 0
        out_.append(tem_input)
    return out_


def loop_check(label_set, input_set):
    set_pairs = []
    for label in label_set:
        for input_label in input_set:
            if len(label_set) > 1 and len(input_set) > 1:
                a_ = copy.deepcopy(label_set)
                a_.remove(label)
                b_ = copy.deepcopy(input_set)
                b_.remove(input_label)
                get_pairs = loop_check(a_, b_)
                for pair in get_pairs:
                    tmp = pair
                    tmp.append([input_label, label])
                    set_pairs.append(tmp)
            elif len(label_set) == 1 and len(input_set) == 1:
                return [[[input_label, label]]]
            else:
                set_pairs.append([[input_label, label]])
    for i in range(len(set_pairs)):
        set_pairs[i].sort()
    temp = []
    for item in set_pairs:
        if item not in temp:
            temp.append(item)
    return temp


def data_reformat(input_data, label):
    max_ = 0
    for label_ in label:
        if label_ > max_:
            max_ = label_
    max_ = max_ + 1
    output_d = []
    for data_ in input_data:
        new_data = []
        for i in range(max_):
            if data_ == i + 1:
                new_data.append(1)
            else:
                new_data.append(0)
        output_d.append(new_data)
    return output_d


def train(epochs, trainLoader, testLoader, model, learning_rate):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    try:
        for e in range(epochs):
            for index, _data in enumerate(tqdm(trainLoader, leave=False)):
                model.train()
                out = model(_data[:-1])
                y_ = _data[-1]
                total_loss = 0
                for i in range(len(out)):
                    loop_set = loop_calculation(out[i], y_[i])
                    min_loss = 999999
                    for data_setting in loop_set:
                        temp_loss = criterion(torch.tensor(data_reformat(data_setting, y_[i]), dtype=torch.float), torch.tensor(y_[i]))
                        if temp_loss < min_loss:
                            min_loss = temp_loss
                    total_loss = total_loss + min_loss
                loss = torch.autograd.Variable(total_loss, requires_grad = True)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                sleep(0.05)
                if index % 20 == 0:
                    print('epoch: {}, batch: {}, loss: {}'.format(e + 1, index + 1, loss.data))
            exec('torch.save(model, os.getcwd() + "//model//" + "model_{}.pt")'.format(e + 1))
            sleep(0.1)
        evaluate_metrics(model=model, test_loader=testLoader)
    except KeyboardInterrupt:
        evaluate_metrics(model=model, test_loader=testLoader)


def demo_work(dataset):
    model_ = torch.load("model.pt")
    test_dataset = dataset
    sleep(0.1)
    evaluate_metrics(model=model_, test_loader=test_dataset)
    print("Among the demo dataset, the results are shown above")


if __name__ == '__main__':
    dataset = []
    for i in range(30): # Change based on your dataset size.
        data = torch.load(osp.join(os.getcwd() + "/data/", 'data_{}.pt'.format(i)))
        dataset.append(data)
    start_training(dataset)
