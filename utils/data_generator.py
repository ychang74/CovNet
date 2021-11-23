import numpy as np
import h5py
import os
import random
import torch
from utils.config import valid_labels
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from sklearn import preprocessing


class JSTSP2021(Dataset):
    def __init__(self, hdf5_path, flusense):
        """Log mel spectrogram of the Dicova Task2 dataset."""
        self.hdf5_path = hdf5_path
        self.flusense = flusense
        self.le = preprocessing.LabelEncoder()
        self.le.fit(valid_labels)

    def __len__(self):
        hf = h5py.File(self.hdf5_path, 'r')
        res = hf['audio_name'].len()
        hf.close()
        return res

    def __getitem__(self, index):
        """Get input and target data of an audio clip.
        """
        with h5py.File(self.hdf5_path, 'r') as hf:
            audio_name = hf['audio_name'][index].decode()
            logmel = hf['logmel'][index]
            if self.flusense:
                label = hf['label'][index].decode()
                label = self.le.transform([label]).item()
            else:
                label = hf['label'][index]
        return {'audio_name': audio_name, 'logmel': logmel, 'label': label}


def dev_fold_dataset(dataset, split_ratio, shuffle, random_seed):
    """Return the train/val dataset for the given fold """
    indices = list(range(len(dataset)))
    data = DataLoader(dataset)
    labels = []
    for d in data:
        labels.append(d['label'].item())
    if len(indices) != len(labels):
        raise Exception('Incorrect train/val split!')

    random.seed(random_seed)
    X = random.sample(indices, k=len(dataset))
    y = labels
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=split_ratio, random_state=random_seed,
                                                      shuffle=shuffle, stratify=y)

    train_indices = [i for i, e in enumerate(X) if e in X_train]
    val_indices = [i for i, e in enumerate(X) if e in X_val]
    if len(train_indices) + len(val_indices) != len(indices) or len(set(train_indices) & set(val_indices)) != 0:
        raise Exception('Incorrect train/val split!')
    '''
    split = int(np.floor(split * len(dataset)))
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    '''
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)

    return train_dataset, val_dataset


def dicova_fold_dataset(dataset, list_dir, fold):
    """Create Dataloaders for dicova"""
    filenames = {}
    pos = 0
    # Iterate the indicated train and val fold
    for fun in ['train', 'val']:
        with open(os.path.join(list_dir, 'dicova_list', '{}_fold_{}.txt'.format(fun, fold))) as f:
            names = f.readlines()
        filenames[fun] = [x.strip() for x in names]

    # Draw from the Dataset according to the given train/val split
    train_indices = []
    val_indices = []
    for i in range(len(dataset)):
        if dataset[i]['audio_name'] in filenames['train']:
            train_indices.append(i)
            if dataset[i]['label'] == 1:
                pos += 1 
        if dataset[i]['audio_name'] in filenames['val']:
            val_indices.append(i)
    if len(train_indices) + len(val_indices) != len(dataset) or len(set(train_indices) & set(val_indices)) != 0:
        raise Exception('Incorrect train/val split!')

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    weights = [pos/len(train_indices), 1- pos/len(train_indices)]

    return train_dataset, val_dataset, weights


def split_compare_dataset(dataset):
    devel_indices = []
    test_indices = []
    train_indices = []

    pos = 0

    for idx in range(len(dataset)):
        if 'devel' in dataset[idx]['audio_name']:
            devel_indices.append(idx)
        elif 'train' in dataset[idx]['audio_name']:
            train_indices.append(idx)
            if dataset[idx]['label'] == 1:
                pos += 1
        else:
            test_indices.append(idx)

    devel_dataset = torch.utils.data.Subset(dataset, devel_indices)
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    weights = [pos / len(train_indices), 1 - pos / len(train_indices)]

    return devel_dataset, train_dataset, test_dataset, weights
