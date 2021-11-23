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
from utils.utilities import create_folder, get_filename, create_logging, Mixup, do_mixup, vote, scoring
from utils.data_generator import JSTSP2021, dev_fold_dataset
from models import BaselineCnn, Vggish, ResNet, MobileNet


def train(args):
    # Arugments & parameters
    workspace = args.workspace
    augmentation = args.augmentation
    pretrain = args.pretrain
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    arch = args.arch
    flusense = args.flusense
    backbone = args.backbone
    num_workers = 8

    # Paths
    logs_dir = os.path.join(workspace, 'logs', '{}'.format(backbone), 'augmentation={}'.format(augmentation),
                            'batch_size={}'.format(batch_size))
    checkpoint_dir = os.path.join(workspace, 'saved_models', '{}'.format(backbone), 'lr-0.001',
                                  'epochs-{}'.format(num_epochs), 'patience-{}'.format(patience))
    create_logging(logs_dir, 'w')
    logging.info(args)
    create_folder(checkpoint_dir)

    if flusense:
        hdf5_path = os.path.join(workspace, 'features_flusense.hdf5')
    else:
        hdf5_path = os.path.join(workspace, 'features.hdf5')

    if 'cuda' in device:
        logging.info('Using GPU.')
    else:
        logging.info('Using CPU. Set --cuda flag to use GPU.')

    if augmentation:
        mixup_augmenter = Mixup(mixup_alpha=1.)

    # Dataset
    dataset = JSTSP2021(hdf5_path=hdf5_path, flusense=flusense)
    # rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=36851234)

    # Train and evaluate
    auc = []
    sensitivity = []
    specificity = []
    time_begin = time.time()
    # for fold, (index_train, index_val) in enumerate(rskf.split(dataset['logmel'], dataset['label'])):
    for cc in range(1):
        logging.info('-' * 30)
        train_dataset, val_dataset = dev_fold_dataset(dataset, split_ratio, shuffle=True, random_seed=random_seed)
        dataloaders = {
            'train': torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                                 num_workers=num_workers),
            'val': torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                               num_workers=num_workers)
        }

        # Model
        if flusense:
            # model = BaselineCnn(classes_num_flusense)
            model = MobileNet(classes_num_flusense)
        else:
            # model = BaselineCnn(classes_num)
            model = MobileNet(classes_num)

        # Parallel
        logging.info('GPU number: {}'.format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)

        if 'cuda' in device:
            model.to(device)

        # Loss function
        if flusense:
            criterion = nn.CrossEntropyLoss(weight=torch.tensor(flusense_weights).to(device))
        else:
            criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 10.0]).to(device))

        # Optimizer
        optimizer = optim.Adam(model.parameters(), lr=learning_rate,
                               betas=(0.9, 0.999), eps=1e-08, weight_decay=0., amsgrad=True)

        # Scheduler
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step, gamma=gamma)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=gamma, patience=patience,
                                                               threshold=0.01, threshold_mode='abs', verbose=True)
        # scores = {}
        for epoch in range(num_epochs):
            logging.info('*' * 10)
            logging.info('Epoch {}/{}, lr:{}'.format(epoch, num_epochs - 1, optimizer.param_groups[0]['lr']))
            logging.info('*' * 10)

            # Each epoch is composed of training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                predicts = []
                truth = []
                file_names = []
                y_scores = []

                # Iterate over data
                for i_batch, sample_batched in enumerate(dataloaders[phase]):
                    if phase == 'train':
                        inputs = sample_batched[arch].to(device)
                        if augmentation:
                            batch_size_curr = len(inputs)
                            labels = sample_batched['label'].type(torch.LongTensor).to(device)
                            batch_lambda = torch.FloatTensor(mixup_augmenter.get_lambda(batch_size=len(inputs))).to(
                                device)
                            mixinputs = do_mixup(inputs, batch_lambda)
                            mixlabels = do_mixup(labels, batch_lambda)
                            # Concatenate the original inputs/labels with mixup inputs/labels for data augmentation
                            inputs = torch.cat([inputs, mixinputs], dim=0)
                            labels = torch.cat([labels, mixlabels], dim=0)
                        else:
                            labels = sample_batched['label'].type(torch.LongTensor).to(device)
                        # forward
                        with torch.set_grad_enabled(phase == 'train'):
                            outputs = model(inputs)
                            # outputs = outputs.squeeze(1)
                            # _, outputs = torch.max(outputs, dim=1)
                            # outputs = outputs.float()
                            loss = criterion(outputs, labels)  # .squeeze(1)
                            s = nn.Softmax(dim=1)
                            outputs = s(outputs)
                            _, preds = torch.max(outputs, dim=1)

                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                            running_loss += loss.item() * inputs.size(0)
                            if augmentation:
                                y_scores.append(outputs.tolist()[:batch_size_curr])
                                predicts.append(preds.tolist()[:batch_size_curr])
                                truth.append(labels.tolist()[:batch_size_curr])
                            else:
                                y_scores.append(outputs.tolist())
                                predicts.append(preds.tolist())
                                truth.append(labels.tolist())
                    else:
                        inputs = sample_batched[arch].to(device)
                        labels = sample_batched['label'].type(torch.LongTensor).to(device)
                        # file_names.append(sample_batched['audio_name'])
                        # forward
                        outputs = model(inputs)
                        # outputs = outputs.squeeze(1)
                        loss = criterion(outputs, labels)  # .squeeze(1)
                        s = nn.Softmax(dim=1)
                        outputs = s(outputs)
                        _, preds = torch.max(outputs, dim=1)

                        running_loss += loss.item() * inputs.size(0)
                        y_scores.append(outputs.tolist())
                        predicts.append(preds.tolist())
                        truth.append(labels.tolist())

                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                y_scores = list(itertools.chain(*y_scores))
                predicts = list(itertools.chain(*predicts))
                truth = list(itertools.chain(*truth))
                # score_epoch = score(truth, predicts)
                confusion_mat, uar, auc = scoring(truth, predicts, y_scores, flusense)
                logging.info(
                    '{}: auc: {:.4f}, uar: {:.4f}, confusion_mat:{}'.format(phase, auc, uar, np.diag(confusion_mat)))
                # if phase == 'val':
                # Record the performance on val in the last epoch
                #    if epoch == num_epochs - 1:
                #        scores['auc'] = score_epoch['AUC']
                #        scores['sensitivity'] = score_epoch['sensitivity']
                #        scores['specificity'] = score_epoch['specificity']

            scheduler.step(uar)
        time_elapsed = time.time() - time_begin
        logging.info('*' * 10)
        logging.info('Training completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

        # Save model
        if not pretrain:
            checkpoint = {
                'model': model.module.state_dict(),
                'optimizer': optimizer.state_dict()}
            if flusense:
                checkpoint_path = os.path.join(checkpoint_dir, 'flusense.pth')
            else:
                checkpoint_path = os.path.join(checkpoint_dir, 'coughvid.pth')
            torch.save(checkpoint, checkpoint_path)
            logging.info('Model saved to {}'.format(checkpoint_path))

        # auc.append(scores['auc'])
        # sensitivity.append(scores['sensitivity'])
        # specificity.append(scores['specificity'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser')
    subparsers = parser.add_subparsers(dest='mode')

    # Train
    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--workspace', type=str, default='../workspace')
    parser_train.add_argument('--augmentation', action='store_true')
    parser_train.add_argument('--pretrain', action='store_true')
    parser_train.add_argument('--flusense', action='store_true')
    parser_train.add_argument('--learning_rate', type=float, default=0.001)
    parser_train.add_argument('--batch_size', type=int, default=16)
    parser_train.add_argument('--arch', type=str, default='logmel')
    parser_train.add_argument('--backbone', type=str, default='baseline')

    # Parse arguments
    args = parser.parse_args()

    if args.mode == 'train':
        train(args)
    else:
        raise Exception('Error argument!')