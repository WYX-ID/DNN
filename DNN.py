# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 16:44:46 2023

@author: YX WANG
"""

import random
from sklearn.metrics import confusion_matrix, roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.io as sio
import numpy as np


seed = 123
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)


# load dataset
data_all = sio.loadmat('DATASET.mat') # DASASET 
X_train = torch.tensor(data_all['trainx'], dtype=torch.float)
y_train = torch.tensor(
    np.argmax(data_all['trainy'], axis=1), dtype=torch.float)
X_test = torch.tensor(data_all['testx'], dtype=torch.float)
y_test = torch.tensor(np.argmax(data_all['testy'], axis=1), dtype=torch.float)
y_train = y_train.view(-1, 1)
y_test = y_test.view(-1, 1)
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
test_dataset = torch.utils.data.TensorDataset(X_test, y_test)


# DNN
class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(15, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 4)
        self.fc4 = nn.Linear(4, 1)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.01)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc4(x)
        x = self.sigmoid(x)
        return x



model = DNN()
optimizer = optim.Adam(model.parameters())
# train the model
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=16, shuffle=True)
for epoch in range(100):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = nn.functional.binary_cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

# test the model
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=32, shuffle=False)
model.eval()
test_loss = 0
correct = 0
y_true = []
y_pred = []
with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        test_loss += nn.functional.binary_cross_entropy(
            output, target, reduction='sum').item()
        pred = output.round()
        correct += pred.eq(target.view_as(pred)).sum().item()
        y_true.extend(target.tolist())
        y_pred.extend(pred.tolist())

test_loss /= len(test_loader.dataset)

print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))