
# coding: utf-8

# In[1]:

import torch
import pandas as pd
import numpy as np
import sys
import os

mutation = sys.argv[1]
seed = int(sys.argv[2])
use_mean = sys.argv[3] == 'True'
net_arch = map(int, sys.argv[4].split(',')) if len(sys.argv) == 5 else []

def get_data(mutation, seed, mean):
    np.random.seed(seed)
    SEQ_PATH = '/data/lisa/data/AML-MILA/sequences.npy'
    SAMPLES_PATH = '/data/lisa/data/AML-MILA/samples.txt'
    LABELS_PATH = '/data/lisa/data/AML-MILA/patients.20170523.txt'
    samples = open(SAMPLES_PATH, 'r').read().split('|')
    labels = pd.read_csv(LABELS_PATH, sep='\t', index_col=0).T
    labels = np.array([int(labels.loc[sample][mutation]) for sample in samples])
    data = np.log(np.load(SEQ_PATH) + 1)

    idx = np.random.permutation(len(data))
    size_train, size_valid = int(len(idx) * 0.6), int(len(idx) * 0.2)
    train_idx, valid_idx = idx[:size_train], idx[size_train:size_train+size_valid]
    if mean:
        train_data = torch.FloatTensor(data[train_idx].mean(axis=1))
        train_label = torch.FloatTensor(labels[train_idx])
    else:
        train_data = torch.FloatTensor(data[train_idx].reshape(-1, data.shape[2]))
        train_label = torch.FloatTensor(np.repeat(labels[train_idx], 100))
    valid_data = torch.FloatTensor(data[valid_idx].reshape(-1, data.shape[2]))
    valid_label = torch.FloatTensor(np.repeat(labels[valid_idx], 100))

    return (train_data, train_label), (valid_data, valid_label)

train_loader, valid_loader = get_data(mutation, seed, use_mean)


# In[9]:

def shuffle_data(data):
    idx = np.random.permutation(len(data[0]))
    train = np.copy(data[0])[idx]
    label = np.copy(data[1])[idx]
    return train, label

def batch_data(data, batch_size):
    return torch.split(data[0], batch_size), torch.split(data[1], batch_size)

# In[35]:

from loader import load_data
import time
import Classifier
import torch.optim as optim
from torch import nn
from torch.autograd import Variable
import sys
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

reload(Classifier)

classifier = Classifier.Classifier(train_loader[0].shape[1], net_arch, False, False, 'ReLU').cuda()
optimizer = optim.Adam(classifier.parameters(), lr=1e-5, betas=(0.5, 0.999))
criterion = nn.BCELoss()
save_path = '/data/milatmp1/lavoiems/IRIC_%s_%s_%s_%s/' % (mutation, net_arch or 'logit', use_mean, seed)
all_train_f1 = []
all_valid_f1 = []
all_train_acc = []
all_valid_acc = []
for i in range(200):
    print('Epoch: %s' % i)
    classifier.train()
    start = time.time()
    shuffle_data(train_loader)
    for data in zip(*batch_data(train_loader, 32)):
        inputs, labels = Variable(data[0].cuda()), Variable(data[1].cuda())
        preds = classifier(inputs)[:,0]
        loss = criterion(preds, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    end = time.time()
    print('Epoch time: %s' % (end - start))
    classifier.eval()

    valid_preds = valid_labels = []
    train_preds = train_labels = []
    for data in zip(*batch_data(valid_loader, 100)):
        inputs, labels = Variable(data[0].cuda()), data[1].numpy()
        preds = classifier(inputs).data.gt_(0.5).cpu().numpy()[:,0]
        valid_preds = np.concatenate([valid_preds, preds])
        valid_labels = np.concatenate([valid_labels, labels])
    for data in zip(*batch_data(train_loader, 32)):
        inputs, labels = Variable(data[0].cuda()), data[1].numpy()
        preds = classifier(inputs).data.gt_(0.5).cpu().numpy()[:,0]
        train_preds = np.concatenate([train_preds, preds])
        train_labels = np.concatenate([train_labels, labels])
    all_valid_f1.append(f1_score(valid_labels, valid_preds))
    print('Validation F1 score: %s' % all_valid_f1[-1])
    all_valid_acc.append(accuracy_score(valid_labels, valid_preds))
    print('Validation Accuracy: %s' % all_valid_acc[-1])
    all_train_f1.append(f1_score(train_labels, train_preds))
    print('Train F1 score: %s' % all_train_f1[-1])
    all_train_acc.append(accuracy_score(train_labels, train_preds))
    print('Train Accuracy: %s' % all_train_acc[-1])

os.mkdir(save_path)
np.save(save_path+'train_f1', all_train_f1)
np.save(save_path+'valid_f1', all_valid_f1)
np.save(save_path+'train_acc', all_train_acc)
np.save(save_path+'valid_acc', all_valid_acc)

# In[5]:


