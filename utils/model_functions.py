import os
import sys

sys.path.append(os.path.join(sys.path[0], '../'))
import torch
import utils.utilities as utt
from utils.config import device, classes_num
import pytorch.embedding as ebd
import pytorch.models as m


def load_train_network(checkpoint_dir, method, backbone, layer):
    map_location = torch.device(device)

    if 'embedding' in method:
        single_model_path = utt.get_model_path(checkpoint_dir, model_name='flusense.pth')
        model = get_embedding_network(backbone, single_model_path, layer[0])
        model_path = utt.get_embedding_model_path(checkpoint_dir, backbone, layer[0])

    elif 'transfer' in method:
        model_path = utt.get_model_path(checkpoint_dir)
        check = torch.load(model_path, map_location)['model']
        check = utt.change_key_name(method, check)
        model = get_network(backbone)
        model.load_state_dict(check)
        trainable_layers = utt.get_trainable_layers(backbone, layer)
        utt.set_parameter_requires_grad(model, trainable_layers)

    else:
        raise Exception('Invalid network method!!')

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
        raise Exception('Invalid embedding network!!')

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
    else:
        raise Exception('Invalid single network!!')
    return model