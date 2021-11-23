import itertools
import os
import sys

sys.path.append(os.path.join(sys.path[0], '../'))
import numpy as np
import argparse
import time
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import RepeatedStratifiedKFold
import torch.utils.data
from sklearn import preprocessing

from utils.config import num_epochs, gamma, step, patience, classes_num, classes_num_flusense, device, random_seed, \
    split_ratio, flusense_weights
from utils.utilities import create_folder, get_filename, create_logging, Mixup, do_mixup, vote, scoring, conv_to_num
from utils.data_generator import JSTSP2021, dev_fold_dataset, dicova_fold_dataset, split_compare_dataset
import embedding as ebd
import transfer
import models as m


def test(args):
    workspace = args.workspace
    batch_size = args.batch_size
    arch = args.arch
    test_dataset = args.test_dataset
    backbone = args.backbone
    method = args.method
    # embedding = args.embedding
    # network = args.network
    learning_rate = args.learning_rate
    layer = args.layer
    num_workers = 8

    # Paths
    logs_dir = os.path.join(workspace, 'logs', '{}'.format(backbone), 'test_data={}'.format(test_dataset),
                            'batch_size={}'.format(batch_size))

    create_logging(logs_dir, 'w')
    logging.info(args)
    # create_folder(checkpoint_dir)

    if test_dataset == 'dicova':
        hdf5_path = os.path.join(workspace, 'features_dicova.hdf5')
    elif test_dataset == 'compare':
        hdf5_path = os.path.join(workspace, 'features_compare.hdf5')
    else:
        raise Exception('Invalid test dataset!!')

    if 'cuda' in device:
        logging.info('Using GPU.')
    else:
        logging.info('Using CPU. Set --cuda flag to use GPU.')

    # Test model
    num = 0
    if 'transfer' in method:
        num = conv_to_num(layer)

    checkpoint_dir = get_checkpoint_dir(method, workspace, backbone, learning_rate, num)
    model = get_test_network(checkpoint_dir, method, backbone, layer)
    # Dataset
    dataset = JSTSP2021(hdf5_path=hdf5_path, flusense=False)

    # Train and evaluate
    aucs = []
    uars = []
    time_begin = time.time()
    # Dicova
    if 'dicova' in test_dataset:
        # Dicova
        for fold in range(1, 6):
            logging.info('-' * 30)
            logging.info('For the {} fold'.format(fold))
            _, val_dataset = dicova_fold_dataset(dataset, workspace, fold)
            dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                                     num_workers=num_workers)

            auc, uar = test_performance(dataloader, arch, model, fold)
            aucs.append(auc)
            uars.append(uar)
    # ComParE21
    if 'compare' in test_dataset:
        logging.info('-' * 30)
        logging.info('{}-embedding_layer-{}-ComParE'.format(backbone, layer))
        compare_devel_dataset, compare_test_dataset = split_compare_dataset(dataset)
        dataloader = torch.utils.data.DataLoader(compare_test_dataset, batch_size=batch_size, shuffle=False,
                                                 num_workers=num_workers)
        auc, uar = test_performance(dataloader, arch, model)

        aucs.append(auc)
        uars.append(uar)

    time_elapsed = time.time() - time_begin
    logging.info('*' * 10)
    logging.info('{} testing completed in {:.0f}m {:.0f}s'.format(test_dataset, time_elapsed // 60, time_elapsed % 60))
    logging.info('the average auc is: {:.4f} and the average uar is: {:.4f}'.format(np.mean(aucs), np.mean(uars)))


def get_checkpoint_dir(method, workspace, backbone, learning_rate=0.001, num=0):
    checkpoint_dir = os.path.join(workspace, 'saved_models', '{}'.format(backbone), 'lr-{}'.format(learning_rate),
                                  'epochs-{}'.format(num_epochs), 'patience-{}'.format(patience))
    if 'transfer' in method:
        checkpoint_dir = os.path.join(workspace, 'transfer_models', '{}'.format(backbone),
                                      'lr-{}'.format(learning_rate), 'ft-{}'.format(num))
    return checkpoint_dir


def get_model_path(checkpoint_dir, model_name='coughvid.pth'):
    model_path = os.path.join(checkpoint_dir, model_name)
    return model_path


def get_embedding_model_path(checkpoint_dir, network, embedding):
    network = network + '_embedding'
    dirpath = os.path.join(checkpoint_dir, 'network-{}'.format(network), 'embedding-{}'.format(embedding))
    model_path = get_model_path(dirpath)
    return model_path


def test_performance(dataloader, arch, model, fold=0):
    y_scores = []
    predicts = []
    truth = []
    for i_batch, sample_batched in enumerate(dataloader):
        inputs = sample_batched[arch].to(device)
        labels = sample_batched['label'].type(torch.LongTensor).to(device)
        outputs = model(inputs)
        s = nn.Softmax(dim=1)
        outputs = s(outputs)
        _, preds = torch.max(outputs, dim=1)

        y_scores.append(outputs.tolist())
        predicts.append(preds.tolist())
        truth.append(labels.tolist())

    y_scores = list(itertools.chain(*y_scores))
    predicts = list(itertools.chain(*predicts))
    truth = list(itertools.chain(*truth))
    confusion_mat, uar, auc = scoring(truth, predicts, y_scores, False)
    logging.info(
        'fold: {}: auc: {:.4f}, uar: {:.4f}, confusion_mat:{}'.format(fold, auc, uar, np.diag(confusion_mat)))
    return auc, uar


def get_test_network(checkpoint_dir, method, backbone, layer):
    map_location = torch.device(device)

    if 'embedding' in method:
        single_model_path = get_model_path(checkpoint_dir, model_name='flusense.pth')
        model = get_embedding_network(backbone, single_model_path, layer[0])
        model_path = get_embedding_model_path(checkpoint_dir, backbone, layer[0])
        model.load_state_dict(torch.load(model_path, map_location)['model'])

    elif 'transfer' in method:
        model_path = get_model_path(checkpoint_dir)
        check = torch.load(model_path, map_location)['model']
        check = change_key_name(method, check)
        model = get_network(backbone)
        model.load_state_dict(check)

    elif 'single' in method:
        model_path = get_model_path(checkpoint_dir)
        model = get_network(backbone)
        model.load_state_dict(torch.load(model_path, map_location)['model'])

    logging.info('GPU number: {}'.format(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model)
    if 'cuda' in device:
        model.to(device)
    model.eval()

    return model


def get_network(backbone):
    if 'baseline' in backbone:
        model = m.BaselineCnn(classes_num)
    elif 'vgg' in backbone:
        model = m.Vggish(classes_num)
    elif 'resnet' in backbone:
        model = m.ResNet(classes_num)
    elif 'mobilenet' in backbone:
        model = m.MobileNet(classes_num)
    return model


def get_embedding_network(backbone, model_path, layer):
    if 'baseline' in backbone:
        model = ebd.BaselineCnnEmbedding(classes_num, model_path, layer)
    elif 'vgg' in backbone:
        model = ebd.VggishEmbedding(classes_num, model_path, layer)
    elif 'resnet' in backbone:
        model = ebd.ResNetEmbedding(classes_num, model_path, layer)
    elif 'mobile' in backbone:
        model = ebd.MobileNetEmbedding(classes_num, model_path, layer)
    else:
        raise Exception('Invalid network!!')

    return model


def change_key_name(mode, state_dict):
    if mode == 'transfer':
        for key in list(state_dict.keys()):
            state_dict[key.replace('model_ft.', '')] = state_dict.pop(key)
    return state_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser')
    subparsers = parser.add_subparsers(dest='mode')

    # Test
    parser_test = subparsers.add_parser('test')
    parser_test.add_argument('--workspace', type=str, default='../../workspace')
    parser_test.add_argument('--batch_size', type=int, default=16)
    parser_test.add_argument('--arch', type=str, default='logmel')
    parser_test.add_argument('--test_dataset', type=str, choices=['dicova', 'compare'], default='dicova')
    parser_test.add_argument('--backbone', type=str, default='baseline')
    parser_test.add_argument('--method', type=str, choices=['transfer', 'embedding', 'single'], default='single')
    # parser_test.add_argument('--embedding', type=str, default='conv1')
    parser_test.add_argument('--learning_rate', type=float, default=0.001)
    parser_test.add_argument('--layer', nargs='*')
    # parser_test.add_argument('--network', type=str, default='baseline_embedding')

    # Parse arguments
    args = parser.parse_args()

    if args.mode == 'test':
        test(args)
    else:
        raise Exception('Error argument!')