import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

import models as m
import constant as c

sys.path.append(os.path.join(sys.path[0], '../'))
from utils.config import device


#################################################################
# Functions

def get_model_dict_by_device(devices, model_path):
    map_location = torch.device(devices)
    return (torch.load(model_path, map_location))['model']

def set_parameter_requires_grad(model, trainable_layers):
    for name, param in model.named_parameters():
        if name not in trainable_layers:
            param.requires_grad = False

def get_trainable_layers(model_name, layer):
    res = []
    pos = c.networks.index(model_name)
    layer.remove('fc')
    if not layer:
        return res
    for l in layer:
        outer = c.outer_layer_name.get(l)[pos]
        if 'block' in outer:
            for inner in c.inner_layer_name.get('block'):
                res.append(outer+inner)
            if 'res' in outer:
                for inner in c.inner_layer_name.get('res'):
                    res.append(outer+inner)
        else:
            res.append(outer+'.weight')
            res.append('bn'+outer[-1]+'.weight')
            res.append('bn'+outer[-1]+'.bias')
    return res


#################################################################
# Baseline Transfer

class BaselineCnnTransfer(nn.Module):
    def __init__(self, classes_num, model_path, layer):
        super(BaselineCnnTransfer, self).__init__()

        load_num = 2 if classes_num == 9 else 9
        self.model_ft = m.BaselineCnn(load_num)

        model_dict = get_model_dict_by_device(device, model_path)
        self.model_ft.load_state_dict(model_dict)

        trainable_layers = get_trainable_layers('baseline', layer)
        print(trainable_layers)
        set_parameter_requires_grad(self.model_ft, trainable_layers)

        num_ftrs = self.model_ft.fc1.in_features
        self.model_ft.fc1 = nn.Linear(num_ftrs, classes_num, bias=True)

        self.init_weights()

    def init_weights(self):
        m.init_layer(self.model_ft.fc1)

    def forward(self, input):
        x = self.model_ft(input)
        return x



###############################################################
# VGG Transfer

class VggishTransfer(nn.Module):
    def __init__(self, classes_num, model_path, layer):
        super(VggishTransfer, self).__init__()

        load_num = 2 if classes_num == 9 else 9
        self.model_ft = m.Vggish(load_num)

        model_dict = get_model_dict_by_device(device, model_path)
        self.model_ft.load_state_dict(model_dict)
        
        trainable_layers = get_trainable_layers('vgg', layer)
        print(trainable_layers)
        set_parameter_requires_grad(self.model_ft, trainable_layers)

        num_ftrs = self.model_ft.fc_final.in_features
        self.model_ft.fc_final = nn.Linear(num_ftrs, classes_num, bias=True)

        self.init_weights()

    def init_weights(self):
        m.init_layer(self.model_ft.fc_final)

    def forward(self, input):
        x = self.model_ft(input)
        return x



#################################################################
# ResNet Transfer

class ResNetTransfer(nn.Module):
    def __init__(self, classes_num, model_path, layer):
        super(ResNetTransfer, self).__init__()
        
        load_num = 2 if classes_num == 9 else 9
        self.model_ft = m.ResNet(load_num)

        model_dict = get_model_dict_by_device(device, model_path)
        self.model_ft.load_state_dict(model_dict)

        trainable_layers = get_trainable_layers('resnet', layer)
        print(trainable_layers)
        set_parameter_requires_grad(self.model_ft, trainable_layers)

        num_ftrs = self.model_ft.fc1.in_features
        self.model_ft.fc1 = nn.Linear(num_ftrs, classes_num, bias=True)
        
        self.init_weights()

    def init_weights(self):
        m.init_layer(self.model_ft.fc1)

    def forward(self, input):
        x = self.model_ft(input)
        return x


#################################################################
# MobileNet Transfer 


class MobileNetTransfer(nn.Module):
    def __init__(self, classes_num, model_path, layer):
        super(MobileNetTransfer, self).__init__()

        load_num = 2 if classes_num == 9 else 9
        self.model_ft = m.MobileNet(load_num)

        model_dict = get_model_dict_by_device(device, model_path)
        self.model_ft.load_state_dict(model_dict)

        trainable_layers = get_trainable_layers('mobilenet', layer)
        print(trainable_layers)
        set_parameter_requires_grad(self.model_ft, trainable_layers)

        num_ftrs = self.model_ft.fc1.in_features
        self.model_ft.fc1 = nn.Linear(num_ftrs, classes_num, bias=True)

        self.init_weights()

        
    def init_weights(self):
        m.init_layer(self.model_ft.fc1)

    def forward(self, input):
        x = self.model_ft(input)
        return x
