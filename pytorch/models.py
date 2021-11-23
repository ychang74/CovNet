import os
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_parameter_requires_grad(model, feature_extracting, trainable_layers):
    if feature_extracting:
        for name, param in model.named_parameters():
            print(name)
            if name not in trainable_layers:
                param.requires_grad = False

def init_layer(layer):
    """Initialize a Linear or Convolutional layer.
    Ref: He, Kaiming, et al. "Delving deep into rectifiers: Surpassing
    human-level performance on imagenet classification." Proceedings of the
    IEEE international conference on computer vision. 2015.
    """

    if layer.weight.ndimension() == 4:
        (n_out, n_in, height, width) = layer.weight.size()
        n = n_in * height * width

    elif layer.weight.ndimension() == 2:
        (n_out, n) = layer.weight.size()

    std = math.sqrt(2. / n)
    scale = std * math.sqrt(3.)
    layer.weight.data.uniform_(-scale, scale)

    if layer.bias is not None:
        layer.bias.data.fill_(0.)

def init_bn(bn):
    """Initialize a Batchnorm layer. """

    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)



#################################################################
# Baseline
class BaselineCnn(nn.Module):
    def __init__(self, class_num):
        super(BaselineCnn, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64,
                               kernel_size=5, stride=1,
                               padding=2, bias=False)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128,
                               kernel_size=5, stride=1,
                               padding=2, bias=False)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256,
                               kernel_size=5, stride=1,
                               padding=2, bias=False)

        self.fc1 = nn.Linear(256, class_num, bias=True)

        self.bn0 = nn.BatchNorm2d(64)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)

        self.init_weights()

    def init_weights(self):
        init_bn(self.bn0)
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_layer(self.conv3)
        init_layer(self.fc1)

        init_bn(self.bn1)
        init_bn(self.bn2)
        init_bn(self.bn3)

    def forward(self, input):
        (_, seq_len, mel_bins) = input.shape
        x = input.view(-1, 1, seq_len, mel_bins)
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
            
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, kernel_size=2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, kernel_size=2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, kernel_size=2)

        x = F.max_pool2d(x, kernel_size=x.shape[2:])
        x = F.dropout(x, p=0.3, training=self.training)
        x = x.view(x.shape[0:2])

        x = self.fc1(x)
        return x



###############################################################
# VGG
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class VggishConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VggishConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=3, stride=1,
                               padding=1, bias=False)

        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=3, stride=1,
                               padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)

        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weights()

    def init_weights(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, input):
        x = input
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, kernel_size=2)

        return x


class Vggish(nn.Module):
    def __init__(self, classes_num):
        super(Vggish, self).__init__()

        self.conv_block1 = VggishConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = VggishConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = VggishConvBlock(in_channels=128, out_channels=256)

        self.bn0 = nn.BatchNorm2d(64)
        self.fc_final = nn.Linear(256, classes_num, bias=True)

        self.init_weights()

    def init_weights(self):
        init_layer(self.fc_final)
        init_bn(self.bn0)

    def forward(self, input):
        (_, seq_len, mel_bins) = input.shape

        x = input.view(-1, 1, seq_len, mel_bins)
        '''(samples_num, feature_maps, time_steps, freq_num)'''
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        x = F.max_pool2d(x, kernel_size=x.shape[2:])
        x = F.dropout(x, p=0.3, training=self.training)
        x = x.view(x.shape[0:2])

        x = self.fc_final(x)

        return x



#################################################################
# ResNet
class ResNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride):
        super(ResNetBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, padding=0, bias=False)
        
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.bn3 = nn.BatchNorm2d(out_ch)
        
        self.init_weights()

    def init_weights(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_layer(self.conv3)
        init_bn(self.bn1)
        init_bn(self.bn2)
        init_bn(self.bn3)

    def forward(self, x):
        identity = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        identity = self.bn3(self.conv3(identity))

        x += identity
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)

        return x
        

class ResNet(nn.Module):
    def __init__(self, class_num):
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.resblock1 = ResNetBlock(64, 128, 1)
        self.resblock2 = ResNetBlock(128, 256, 2)

        self.fc1 = nn.Linear(256, class_num, bias=True)

        self.bn0 = nn.BatchNorm2d(64)

        self.init_weights()

    def init_weights(self):
        init_bn(self.bn0)
        init_layer(self.conv1)
        init_layer(self.fc1)
        init_bn(self.bn1)

    def forward(self, input):
        (_, seq_len, mel_bins) = input.shape
        x = input.view(-1, 1, seq_len, mel_bins)
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
            
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)

        x = self.resblock1(x)
        x = self.resblock2(x)
        
        x = F.max_pool2d(x, kernel_size=x.shape[2:])
        x = F.dropout(x, p=0.3, training=self.training)
        x = x.view(x.shape[0:2])

        x = self.fc1(x)
        return x


#################################################################
# MobileNet
class MobileNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super(MobileNetBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride, padding=1, groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        
        self.conv2 = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        
        self.init_weights()
    
    def init_weights(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x
        

class MobileNet(nn.Module):
    def __init__(self, class_num):
        super(MobileNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, kernel_size =3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.mobileblock1 = MobileNetBlock(64, 128, 1)
        self.mobileblock2 = MobileNetBlock(128, 256, 2)

        self.fc1 = nn.Linear(256, class_num, bias=True)
        self.bn0 = nn.BatchNorm2d(64)

        self.init_weights()

        
    def init_weights(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.conv1)
        init_bn(self.bn1)

    def forward(self, input):
        (_, seq_len, mel_bins) = input.shape
        x = input.view(-1, 1, seq_len, mel_bins)
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
            
        x = F.relu(self.bn1(self.conv1(x)))

        x = self.mobileblock1(x)
        x = self.mobileblock2(x)
        
        x = F.max_pool2d(x, kernel_size=x.shape[2:])
        x = F.dropout(x, p=0.15, training=self.training)
        x = x.view(x.shape[0:2])

        x = self.fc1(x)
        return x
