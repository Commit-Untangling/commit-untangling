from main import demo_work
import numpy as np
import torch
import os


dataset = []
root = os.getcwd()
print(root)
for i in range(5):
    data = torch.load(root + '\\data\\data_{}.pt'.format(i))
    dataset.append(data)
demo_work(dataset)

