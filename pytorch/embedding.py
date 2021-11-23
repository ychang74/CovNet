import torch
import torch.nn as nn
import torch.nn.functional as F

import models as m
import constant as c
import sys
import os

sys.path.append(os.path.join(sys.path[0], '../'))
from utils.config import device

###############################################################
# Functions

layer_val = {}


def hook_layer_value(name):
    def hook(model, input, output):
        layer_val[name] = output.detach()

    return hook


def get_layer_paras(embedding_layer_key):
    return c.layer_paras[embedding_layer_key]


def get_model_dict_by_device(devices, model_path):
    map_location = torch.device(devices)
    return (torch.load(model_path, map_location))['model']


def map_layer_name(embedding_layer, is_plus_embedding=False):
    if is_plus_embedding:
        return 'default'
    for key, values in c.layer_name.items():
        if embedding_layer in values:
            return key


def is_plus_embedding(embedding_layer):
    if '+' in embedding_layer:
        return True
    return False


def embedding_layer(x, x1, is_plus_embedding):
    if is_plus_embedding:
        x = torch.add(x, x1)
        return x
    x = torch.cat([x, x1], dim=1)
    return x


###############################################################
# BaselineCnn Embedding

class BaselineCnnEmbedding(nn.Module):
    def __init__(self, class_num, model_path, embedding_layer):
        super(BaselineCnnEmbedding, self).__init__()

        self.is_plus_embedding = is_plus_embedding(embedding_layer)
        self.embedding_layer = embedding_layer.replace("+", '')

        load_num = 2 if class_num == 9 else 9
        self.model_ft = m.BaselineCnn(load_num)

        model_dict = get_model_dict_by_device(device, model_path)
        self.model_ft.load_state_dict(model_dict)

        embedding_layer_key = map_layer_name(self.embedding_layer, self.is_plus_embedding)
        layer_paras = get_layer_paras(embedding_layer_key)

        num_ftrs = self.model_ft.fc1.in_features
        self.model_ft.fc1 = nn.Linear(num_ftrs, class_num, bias=True)

        self.conv1 = nn.Conv2d(in_channels=layer_paras[0], out_channels=layer_paras[1],
                               kernel_size=5, stride=1,
                               padding=2, bias=False)

        self.conv2 = nn.Conv2d(in_channels=layer_paras[3], out_channels=layer_paras[4],
                               kernel_size=5, stride=1,
                               padding=2, bias=False)

        self.conv3 = nn.Conv2d(in_channels=layer_paras[6], out_channels=layer_paras[7],
                               kernel_size=5, stride=1,
                               padding=2, bias=False)

        self.fc1 = nn.Linear(layer_paras[8], class_num, bias=True)

        self.bn0 = nn.BatchNorm2d(64)
        self.bn1 = nn.BatchNorm2d(layer_paras[2])
        self.bn2 = nn.BatchNorm2d(layer_paras[5])
        self.bn3 = nn.BatchNorm2d(layer_paras[8])

        self.init_weights()

    def get_layer_value(self, inputs):
        if 'conv1' in self.embedding_layer:
            self.model_ft.conv1.register_forward_hook(hook_layer_value('conv1'))
        elif 'conv2' in self.embedding_layer:
            self.model_ft.conv2.register_forward_hook(hook_layer_value('conv2'))
        elif 'conv3' in self.embedding_layer:
            self.model_ft.conv3.register_forward_hook(hook_layer_value('conv3'))

        output = self.model_ft(inputs)
        return layer_val[self.embedding_layer]

    def init_weights(self):
        m.init_bn(self.bn0)
        m.init_layer(self.conv1)
        m.init_layer(self.conv2)
        m.init_layer(self.conv3)
        m.init_layer(self.fc1)

        m.init_bn(self.bn1)
        m.init_bn(self.bn2)
        m.init_bn(self.bn3)

    def forward(self, inputs):
        # get value of conv#
        x1 = self.get_layer_value(inputs)

        (_, seq_len, mel_bins) = inputs.shape
        x = inputs.view(-1, 1, seq_len, mel_bins)
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        # Conv1 embedding
        x = self.conv1(x)
        if 'conv1' in self.embedding_layer:
            x = embedding_layer(x1, x, self.is_plus_embedding)

        x = F.relu(self.bn1(x))
        x = F.max_pool2d(x, kernel_size=2)

        # Conv2 embedding
        x = self.conv2(x)
        if 'conv2' == self.embedding_layer:
            x = embedding_layer(x1, x, self.is_plus_embedding)

        x = F.relu(self.bn2(x))
        x = F.max_pool2d(x, kernel_size=2)

        # Conv3 embedding
        x = self.conv3(x)
        if 'conv3' == self.embedding_layer:
            x = embedding_layer(x1, x, self.is_plus_embedding)

        x = F.relu(self.bn3(x))
        x = F.max_pool2d(x, kernel_size=2)

        x = F.max_pool2d(x, kernel_size=x.shape[2:])
        x = F.dropout(x, p=0.3, training=self.training)
        x = x.view(x.shape[0:2])

        x = self.fc1(x)
        return x


###############################################################
# VGG Embedding

class VggishEmbedding(nn.Module):
    def __init__(self, classes_num, model_path, embedding_layer):
        super(VggishEmbedding, self).__init__()

        self.is_plus_embedding = is_plus_embedding(embedding_layer)
        self.embedding_layer = embedding_layer.replace("+", '')

        load_num = 2 if classes_num == 9 else 9
        self.model_ft = m.Vggish(load_num)

        model_dict = get_model_dict_by_device(device, model_path)
        self.model_ft.load_state_dict(model_dict)

        embedding_layer_key = map_layer_name(self.embedding_layer, self.is_plus_embedding)
        layer_paras = get_layer_paras(embedding_layer_key)

        self.conv_block1 = m.VggishConvBlock(in_channels=layer_paras[0], out_channels=layer_paras[1])
        self.conv_block2 = m.VggishConvBlock(in_channels=layer_paras[3], out_channels=layer_paras[4])
        self.conv_block3 = m.VggishConvBlock(in_channels=layer_paras[6], out_channels=layer_paras[7])

        self.bn0 = nn.BatchNorm2d(64)
        self.fc_final = nn.Linear(layer_paras[8], classes_num, bias=True)

        self.init_weights()

    def get_layer_value(self, inputs):

        if 'conv_block1' in self.embedding_layer:
            self.model_ft.conv_block1.register_forward_hook(hook_layer_value('conv_block1'))
        elif 'conv_block2' in self.embedding_layer:
            self.model_ft.conv_block2.register_forward_hook(hook_layer_value('conv_block2'))
        elif 'conv_block3' in self.embedding_layer:
            self.model_ft.conv_block3.register_forward_hook(hook_layer_value('conv_block3'))

        output = self.model_ft(inputs)
        return layer_val[self.embedding_layer]

    def init_weights(self):
        m.init_layer(self.fc_final)
        m.init_bn(self.bn0)

    def forward(self, inputs):

        x1 = self.get_layer_value(inputs)

        (_, seq_len, mel_bins) = inputs.shape
        x = inputs.view(-1, 1, seq_len, mel_bins)
        '''(samples_num, feature_maps, time_steps, freq_num)'''
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        x = self.conv_block1(x)
        # embedding conv_block1
        if 'conv_block1' == self.embedding_layer:
            x = embedding_layer(x1, x, self.is_plus_embedding)

        x = self.conv_block2(x)
        # embedding conv_block2
        if 'conv_block2' in self.embedding_layer:
            x = embedding_layer(x1, x, self.is_plus_embedding)

        x = self.conv_block3(x)
        # embedding conv_block3
        if 'conv_block3' == self.embedding_layer:
            x = embedding_layer(x1, x, self.is_plus_embedding)

        x = F.max_pool2d(x, kernel_size=x.shape[2:])
        x = F.dropout(x, p=0.3, training=self.training)
        x = x.view(x.shape[0:2])

        x = self.fc_final(x)

        return x


#################################################################
# ResNet Embedding
class ResNetEmbedding(nn.Module):
    def __init__(self, classes_num, model_path, embedding_layer):
        super(ResNetEmbedding, self).__init__()

        self.is_plus_embedding = is_plus_embedding(embedding_layer)
        self.embedding_layer = embedding_layer.replace("+", '')

        load_num = 2 if classes_num == 9 else 9
        self.model_ft = m.ResNet(load_num)

        model_dict = get_model_dict_by_device(device, model_path)
        self.model_ft.load_state_dict(model_dict)

        embedding_layer_key = map_layer_name(self.embedding_layer, self.is_plus_embedding)
        layer_paras = get_layer_paras(embedding_layer_key)

        self.conv1 = nn.Conv2d(layer_paras[0], layer_paras[1], kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(layer_paras[2])

        self.resblock1 = m.ResNetBlock(layer_paras[3], layer_paras[4], 1)
        self.resblock2 = m.ResNetBlock(layer_paras[6], layer_paras[7], 2)

        self.fc1 = nn.Linear(layer_paras[8], classes_num, bias=True)

        self.bn0 = nn.BatchNorm2d(64)

        self.init_weights()

    def get_layer_value(self, inputs):
        if 'conv1' in self.embedding_layer:
            self.model_ft.conv1.register_forward_hook(hook_layer_value('conv1'))
        elif 'resblock1' in self.embedding_layer:
            self.model_ft.resblock1.register_forward_hook(hook_layer_value('resblock1'))
        elif 'resblock2' in self.embedding_layer:
            self.model_ft.resblock2.register_forward_hook(hook_layer_value('resblock2'))

        output = self.model_ft(inputs)
        return layer_val[self.embedding_layer]

    def init_weights(self):
        m.init_bn(self.bn0)
        m.init_layer(self.conv1)
        m.init_layer(self.fc1)
        m.init_bn(self.bn1)

    def forward(self, inputs):

        x1 = self.get_layer_value(inputs)

        (_, seq_len, mel_bins) = inputs.shape
        x = inputs.view(-1, 1, seq_len, mel_bins)
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        x = self.conv1(x)
        if 'conv1' in self.embedding_layer:
            x = embedding_layer(x1, x, self.is_plus_embedding)
        x = F.relu(self.bn1(x))

        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)

        x = self.resblock1(x)
        if 'resblock1' in self.embedding_layer:
            x = embedding_layer(x1, x, self.is_plus_embedding)

        x = self.resblock2(x)
        if 'resblock2' in self.embedding_layer:
            x = embedding_layer(x1, x, self.is_plus_embedding)

        x = F.max_pool2d(x, kernel_size=x.shape[2:])
        x = F.dropout(x, p=0.3, training=self.training)
        x = x.view(x.shape[0:2])

        x = self.fc1(x)
        return x


#################################################################
# MobileNet Embedding

class MobileNetEmbedding(nn.Module):
    def __init__(self, classes_num, model_path, embedding_layer):
        super(MobileNetEmbedding, self).__init__()

        self.is_plus_embedding = is_plus_embedding(embedding_layer)
        self.embedding_layer = embedding_layer.replace("+", '')

        load_num = 2 if classes_num == 9 else 9
        self.model_ft = m.MobileNet(load_num)

        model_dict = get_model_dict_by_device(device, model_path)
        self.model_ft.load_state_dict(model_dict)

        embedding_layer_key = map_layer_name(self.embedding_layer, self.is_plus_embedding)
        layer_paras = get_layer_paras(embedding_layer_key)

        self.conv1 = nn.Conv2d(layer_paras[0], layer_paras[1], kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(layer_paras[2])

        self.mobileblock1 = m.MobileNetBlock(layer_paras[3], layer_paras[4], 1)
        self.mobileblock2 = m.MobileNetBlock(layer_paras[6], layer_paras[7], 2)

        self.fc1 = nn.Linear(layer_paras[8], classes_num, bias=True)
        self.bn0 = nn.BatchNorm2d(64)

        self.init_weights()

    def get_layer_value(self, inputs):

        if 'conv1' in self.embedding_layer:
            self.model_ft.conv1.register_forward_hook(hook_layer_value('conv1'))
        elif 'mobileblock1' in self.embedding_layer:
            self.model_ft.mobileblock1.register_forward_hook(hook_layer_value('mobileblock1'))
        elif 'mobileblock2' in self.embedding_layer:
            self.model_ft.mobileblock2.register_forward_hook(hook_layer_value('mobileblock2'))

        output = self.model_ft(inputs)
        return layer_val[self.embedding_layer]

    def init_weights(self):
        m.init_bn(self.bn0)
        m.init_layer(self.fc1)
        m.init_layer(self.conv1)
        m.init_bn(self.bn1)

    def forward(self, inputs):

        x1 = self.get_layer_value(inputs)

        (_, seq_len, mel_bins) = inputs.shape
        x = inputs.view(-1, 1, seq_len, mel_bins)
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        x = self.conv1(x)
        if 'conv1' in self.embedding_layer:
            x = embedding_layer(x1, x, self.is_plus_embedding)
        x = F.relu(self.bn1(x))

        x = self.mobileblock1(x)
        if 'mobileblock1' in self.embedding_layer:
            x = embedding_layer(x1, x, self.is_plus_embedding)

        x = self.mobileblock2(x)
        if 'mobileblock2' in self.embedding_layer:
            x = embedding_layer(x1, x, self.is_plus_embedding)

        x = F.max_pool2d(x, kernel_size=x.shape[2:])
        x = F.dropout(x, p=0.15, training=self.training)
        x = x.view(x.shape[0:2])

        x = self.fc1(x)
        return x
