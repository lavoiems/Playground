from torchvision import datasets, transforms
import torch
import pandas as pd
import numpy as np


def load_data(mutation, seed, batch_size):
    SEQ_PATH = '/data/lisa/data/AML-MILA/sequences.npy'
    SAMPLES_PATH = '/data/lisa/data/AML-MILA/samples.txt'
    LABELS_PATH = '/data/lisa/data/AML-MILA/patients.20170523.txt'
    samples = open(SAMPLES_PATH, 'r').read().split('|')
    labels = pd.read_csv(LABELS_PATH, sep='\t', index_col=0).T
    labels = [int(labels.loc[sample][mutation]) for sample in samples][:30]
    data = np.log(np.load(SEQ_PATH)[:30] + 1)

    np.random.seed(seed)
    idx = np.random.permutation(len(data))
    size_train, size_valid = int(len(idx) * 0.6), int(len(idx) * 0.2)
    train_idx, valid_idx = idx[:size_train], idx[size_train:size_train+size_valid]
    train_data = torch.FloatTensor(data[train_idx].reshape(-1, data.shape[2]))
    valid_data = torch.FloatTensor(data[valid_idx].reshape(-1, data.shape[2]))
    train_label = torch.FloatTensor(np.repeast(labels[train_idx], 100))
    valid_label = torch.FloatTensor(np.repeat(labels[valid_idx], 100))

    train_data = torch.split(train_data, batch_size)
    valid_data = torch.split(valid_data, batch_size)
    train_label = torch.split(train_label, batch_size)
    valid_label = torch.split(valid_label, batch_size)
    return (train_data, train_label) (valid_data, valid_label)


