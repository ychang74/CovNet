import numpy as np
import soundfile
import librosa
import os
from sklearn import metrics
from collections import Counter
import logging
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from collections import OrderedDict
from sklearn.metrics import recall_score, confusion_matrix, roc_auc_score


def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)


def get_filename(path):
    path = os.path.realpath(path)
    name_ext = path.split('/')[-1]
    name = os.path.splitext(name_ext)[0]
    return name


def create_logging(log_dir, filemode):
    create_folder(log_dir)
    i1 = 0

    while os.path.isfile(os.path.join(log_dir, '%04d.log' % i1)):
        i1 += 1

    log_path = os.path.join(log_dir, '%04d.log' % i1)
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
        datefmt='%a, %d %b %Y %H:%M:%S',
        filename=log_path,
        filemode=filemode)

    # Print to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    return logging


def read_audio(path, max_len, target_fs=None):
    (audio, fs) = soundfile.read(path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if target_fs is not None and fs != target_fs:
        audio = librosa.resample(audio, orig_sr=fs, target_sr=target_fs)
        fs = target_fs

    if len(audio) < max_len:
        n = math.ceil(max_len / len(audio))
        audio = np.tile(audio, n)

    return audio[:max_len], fs


def read_audio_flusense(path, start, end, target_fs=None):
    start_frame = math.ceil(44100 * start)
    stop_frame = math.floor(44100 * end)

    (audio, fs) = soundfile.read(path, start=start_frame, stop=stop_frame)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if target_fs is not None and fs != target_fs:
        audio = librosa.resample(audio, orig_sr=fs, target_sr=target_fs)
        fs = target_fs
    
    return audio, fs


def write_predictions(filenames_unique, predicts_unique, truth_csv, output_csv):
    if 'test' in truth_csv:
        with open(truth_csv, 'r') as f:
            names = f.readlines()[1:]
        filenames_truth = [x.strip().split(',')[0] for x in names]
        f.close()
    else:
        with open(truth_csv, 'r') as f:
            names = f.readlines()
        filenames_truth = [x.strip() for x in names]
        f.close()

    f = open(output_csv, 'a')
    for i in range(0, len(filenames_truth)):
        filename_curr = filenames_truth[i]
        ind = filenames_unique.index(filename_curr)
        preds_curr = predicts_unique[ind]
        f.write(filename_curr + ' ' + str(preds_curr) + '\n')

    f.close()


def scoring(truth, pred, y_scores, flusense):
    y_scores = np.asarray(y_scores)
    uar = recall_score(truth, pred, average='macro')
    if flusense:
        auc = roc_auc_score(truth, y_scores, average='macro', multi_class='ovo')
        confusion_mat = confusion_matrix(truth, pred, labels=list(range(9)))
    else:
        auc = roc_auc_score(truth, y_scores[:, 1], average='macro')
        confusion_mat = confusion_matrix(truth, pred, labels=list(range(2)))
    return confusion_mat, uar, auc
    


class Mixup(object):
    """ https://github.com/qiuqiangkong/audioset_tagging_cnn/blob/750c318c0fcf089bd430f4d58e69451eec55f0a9/pytorch/pytorch_utils.py#L18
    """

    def __init__(self, mixup_alpha, random_seed=1234):
        """Mixup coefficient generator.
        """
        self.mixup_alpha = mixup_alpha
        self.random_state = np.random.RandomState(random_seed)

    def get_lambda(self, batch_size):
        """Get mixup random coefficients.
        Args:
          batch_size: int
        Returns:
          mixup_lambdas: (batch_size,)
        """
        mixup_lambdas = []
        for n in range(0, batch_size, 2):
            lam = self.random_state.beta(self.mixup_alpha, self.mixup_alpha, 1)[0]
            mixup_lambdas.append(lam)
            mixup_lambdas.append(1. - lam)

        if not batch_size % 2 == 0:
            mixup_lambdas = mixup_lambdas[:-1]
        return np.array(mixup_lambdas)


def do_mixup(x, mixup_lambda):
    """ https://github.com/qiuqiangkong/audioset_tagging_cnn/blob/750c318c0fcf089bd430f4d58e69451eec55f0a9/pytorch/pytorch_utils.py#L18
    Mixup x of even indexes (0, 2, 4, ...) with x of odd indexes
    (1, 3, 5, ...).
    Args:
      x: (batch_size * 2, ...)
      mixup_lambda: (batch_size * 2,)
    Returns:
      out: (batch_size, ...)
    """
    if len(x) % 2 == 0:
        out = (x[0:: 2].transpose(0, -1) * mixup_lambda[0:: 2] + \
            x[1:: 2].transpose(0, -1) * mixup_lambda[1:: 2]).transpose(0, -1)
    else:
        out = (x[0:-1: 2].transpose(0, -1) * mixup_lambda[0:-1: 2] + \
               x[1:: 2].transpose(0, -1) * mixup_lambda[1:: 2]).transpose(0, -1)
    return out


def vote(predicts, strategy):
    """
    Return the fused predicts from three modalities based on the strategy
    :param predicts: list of three predicts from three modalities
    :param strategy: max or mean
    :return: fused predicts
    """
    predict = []
    for i in range(len(predicts[0])):
        if strategy == 'mean':
            predict.append(np.mean([predicts[0][i], predicts[1][i], predicts[2][i]]))
        elif strategy == 'max':
            predict.append(np.max([predicts[0][i], predicts[1][i], predicts[2][i]]))
    return predict


def conv_to_num(conv):
    convs = [['fc'], ['fc', 'conv3'], ['fc', 'conv2', 'conv3'], ['fc', 'conv1', 'conv2', 'conv3']]
    if conv not in convs:
        raise Exception('Incorrect layer!')
    return convs.index(conv)
