import random
from time import sleep

import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch import nn
from tqdm import tqdm
import model
import os.path as osp
import os


def start_training(dataset):
    random.shuffle(dataset)
    train_dataset = dataset[:25]
    test_dataset = dataset[25:]
    model_ = model.Utango(h_size=64, max_method=3, drop_out_rate=0.5, gcn_layers=3)
    train(epochs=10, trainLoader=train_dataset, testLoader=test_dataset, model=model_, learning_rate=0.0001)


def evaluate_metrics(model, test_loader):
    model.eval()
    with torch.no_grad():
        hit = [0, 0, 0, 0, 0, 0]
        for data in tqdm(test_loader):
            correct = 0
            total = 0
            _, out = model(data)
            print(out)
            print(data.y)

def train(epochs, trainLoader, testLoader, model, learning_rate):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    try:
        for e in range(epochs):
            for index, _data in enumerate(tqdm(trainLoader, leave=False)):
                out = model(_data)
                y_1 = torch.reshape(torch.tensor(_data.y[1]), (-1,))
                y_2 = torch.reshape(torch.tensor(_data.y[2]), (-1,))
                y_3 = torch.reshape(torch.tensor(_data.y[3]), (-1,))
                y_4 = torch.reshape(torch.tensor(_data.y[4]), (-1,))
                y_5 = torch.reshape(torch.tensor(_data.y[5]), (-1,))
                y_6 = torch.reshape(torch.tensor(_data.y[6]), (-1,))
                y_7 = torch.reshape(torch.tensor(_data.y[7]), (-1,))
                out_1 = out[0].clone().detach().requires_grad_(True)
                out_2 = out[1].clone().detach().requires_grad_(True)
                out_3 = out[2].clone().detach().requires_grad_(True)
                out_4 = out[3].clone().detach().requires_grad_(True)
                out_5 = out[4].clone().detach().requires_grad_(True)
                out_6 = out[5].clone().detach().requires_grad_(True)
                out_7 = out[6].clone().detach().requires_grad_(True)
                loss = torch.autograd.Variable((criterion(out_1, y_1) + criterion(out_2, y_2) + criterion(out_3, y_3) + criterion(out_4, y_4) + criterion(out_5, y_5) + criterion(out_6, y_6) + criterion(out_7, y_7))/7, requires_grad = True)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                sleep(0.05)
                if index % 20 == 0:
                    print('epoch: {}, batch: {}, loss: {}'.format(e + 1, index + 1, loss.data))
            evaluate_metrics(model=model, test_loader=testLoader)
            sleep(0.1)
    except KeyboardInterrupt:
        evaluate_metrics(model=model, test_loader=testLoader)


if __name__ == '__main__':
    dataset = []
    for i in range(30):
        data = torch.load(osp.join(os.getcwd() + "\\data\\", 'data_{}.pt'.format(i)))
        dataset.append(data)
    start_training(dataset)
