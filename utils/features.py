import os
import sys
import numpy as np
import pandas as pd
import argparse
import librosa
import math 
from scipy import signal
import time
from utilities import read_audio, read_audio_flusense, create_folder
import config
import soundfile
import h5py

sys.path.insert(1, os.path.join(sys.path[0], '../utils'))


class LogMelExtractor:
    def __init__(self, sample_rate, window_size, overlap, mel_bins):
        self.window_size = window_size
        self.overlap = overlap
        self.ham_win = np.hamming(window_size)

        self.melW = librosa.filters.mel(sr=sample_rate,
                                        n_fft=window_size,
                                        n_mels=mel_bins,
                                        fmin=20.,
                                        fmax=sample_rate // 2).T

    def transform(self, audio):
        ham_win = self.ham_win
        window_size = self.window_size
        overlap = self.overlap

        [_, _, x] = signal.spectral.spectrogram(
            audio,
            window=ham_win,
            nperseg=window_size,
            noverlap=overlap,
            detrend=False,
            return_onesided=True,
            mode='magnitude')
        x = x.T

        x = np.dot(x, self.melW)
        x = np.log(x + 1e-8)
        x = x.astype(np.float32)

        return x


def calculate_logmel(audio_path, sample_rate, feature_extractor):
    # Read audio
    (audio, fs) = read_audio(audio_path, max_len=config.audio_samples, target_fs=sample_rate)

    '''We do not divide the maximum value of an audio here because we assume 
    the low energy of an audio may also contain information of a scene. '''

    # Extract feature
    feature = feature_extractor.transform(audio)

    return feature


def read_meta(meta_csv):
    df = pd.read_csv(meta_csv)
    df = pd.DataFrame(df)

    audio_names = []
    covid_statuses = []
    genders = []

    for row in df.iterrows():
        file_name = row[1]['uuid']
        covid_status = row[1]['status']
        covid_status = 0 if covid_status == 'healthy' else 1
        gender = row[1]['gender']

        audio_names.append(file_name)
        covid_statuses.append(covid_status)
        genders.append(gender)

    return audio_names, covid_statuses, genders


def read_meta_flusense(meta_csv, exclude):
    df = pd.read_csv(meta_csv)
    df = pd.DataFrame(df)

    audio_names = []
    labels = []
    intervals = []
    count = 0 

    for row in df.iterrows():
        file_name = row[1]['filename']
        label = row[1]['label']
        interval = [row[1]['start'], row[1]['end']]
       
        # Skip the audio intervals with length smaller than 0.5s
        # Skip the audios with labels in the exclude (list)
        dura = interval[1] - interval[0]
        if dura < 0.5 or label in exclude:
            continue
        c1, c2 = divmod(dura, 1)
        count += c1
        if c2 >= 0.5:
            count += 1

        audio_names.append(file_name)
        labels.append(label)
        intervals.append(interval)

    return audio_names, labels, intervals, int(count)


def read_meta_dicova(meta_csv):
    df = pd.read_csv(meta_csv)
    df = pd.DataFrame(df)

    audio_names = []
    covid_statuses = []
    genders = []

    for row in df.iterrows():
        file_name = row[1]['File_name']
        covid_status = row[1]['Covid_status']
        covid_status = 0 if covid_status == 'n' else 1
        gender = row[1]['Gender']

        audio_names.append(file_name)
        covid_statuses.append(covid_status)
        genders.append(gender)

    return audio_names, covid_statuses, genders


def calculate_features(args):
    # Arguments & parameters
    dataset_dir = args.dataset_dir
    workspace = args.workspace
    dataset = args.dataset
    
    if not os.path.exists(workspace):
        os.makedirs(workspace)

    sample_rate = config.sample_rate
    audio_samples = config.audio_samples
    audio_samples_flusense = config.audio_samples_flusense
    window_size = config.window_size
    overlap = config.overlap
    mel_bins = config.mel_bins

    # Feature extractor
    feature_extractor = LogMelExtractor(sample_rate=sample_rate,
                                        window_size=window_size,
                                        overlap=overlap,
                                        mel_bins=mel_bins)

    # Read meta csv
    if dataset == 'flusense':
        audio_dir = os.path.join(dataset_dir, 'audio_flusense')
        meta_csv = os.path.join(dataset_dir, 'meta_flusense.csv')
        audio_names, labels, intervals, seg_num = read_meta_flusense(meta_csv, config.exclude)
        hdf5_path = os.path.join(workspace, 'features_flusense.hdf5')
    elif dataset == 'coughvid':
        audio_dir = os.path.join(dataset_dir, 'audio')
        meta_csv = os.path.join(dataset_dir, 'meta.csv')
        [audio_names, covid_statuses, _] = read_meta(meta_csv)
        hdf5_path = os.path.join(workspace, 'features.hdf5')
    elif dataset == 'dicova':
        audio_dir = os.path.join(dataset_dir, 'audio_dicova')
        meta_csv = os.path.join(dataset_dir, 'meta_dicova.csv')
        [audio_names, covid_statuses, _] = read_meta_dicova(meta_csv)
        hdf5_path = os.path.join(workspace, 'features_dicova.hdf5')
    
    audios_num = len(audio_names)
    print('#audios in meta csv: {}'.format(audios_num))

    if os.path.exists('temp.wav'):
        os.remove('temp.wav')
        #time.sleep(1)

    # Read audios and extract features
    since_time = time.time()
    with h5py.File(hdf5_path, 'w') as hf:
        if dataset == 'flusense':
            hf.create_dataset(name='audio_name', shape=(seg_num,), dtype='S80')
            hf.create_dataset(name='logmel', shape=(seg_num, audio_samples/overlap - 1, mel_bins), dtype=np.float32)
            hf.create_dataset(name='label', shape=(seg_num,), dtype='S80')
            count = 0
            for audio_name, label, interval in list(zip(audio_names, labels, intervals)):
                if not count % 1000:
                    print('Have processed {} segments'.format(count))
                audio_path = os.path.join(audio_dir, audio_name + '.wav')
                (audio, fs) = read_audio_flusense(audio_path, start=interval[0], end=interval[1], target_fs=sample_rate)
                if len(audio) <= audio_samples_flusense:
                    n = math.ceil(audio_samples_flusense / len(audio))
                    audio = np.tile(audio, n)[:audio_samples_flusense]
                    soundfile.write(file='temp.wav', data=audio, samplerate=fs)
                    feature = calculate_logmel(audio_path='temp.wav',
                                               sample_rate=sample_rate,
                                               feature_extractor=feature_extractor)
                    hf['audio_name'][count] = audio_name.encode()
                    hf['label'][count] = label.encode()
                    hf['logmel'][count] = feature
                    count += 1
                    os.remove('temp.wav')
                    #time.sleep(1)
                else:
                    for seg in range(0, len(audio) - audio_samples_flusense + 1, audio_samples_flusense):
                        soundfile.write(file='temp.wav', data=audio[seg:(seg+audio_samples_flusense)], samplerate=fs)
                        feature = calculate_logmel(audio_path='temp.wav',
                                                   sample_rate=sample_rate,
                                                   feature_extractor=feature_extractor)
                        hf['audio_name'][count] = audio_name.encode()
                        hf['label'][count] = label.encode()
                        hf['logmel'][count] = feature
                        count += 1 
                        os.remove('temp.wav')
                        #time.sleep(1)
                    if len(audio) - (seg + audio_samples_flusense) >= audio_samples_flusense / 2.0:
                        audio = np.tile(audio[seg+audio_samples_flusense:], 2)[:audio_samples_flusense]
                        soundfile.write(file='temp.wav', data=audio, samplerate=fs)
                        feature = calculate_logmel(audio_path='temp.wav',
                                                   sample_rate=sample_rate,
                                                   feature_extractor=feature_extractor)
                        hf['audio_name'][count] = audio_name.encode()
                        hf['label'][count] = label.encode()
                        hf['logmel'][count] = feature
                        count += 1 
                        os.remove('temp.wav')
                        #time.sleep(1)
            print('There are total {} segments in the flusense hdf5 file'.format(count))
        
        elif dataset == 'coughvid':
            hf.create_dataset(name='audio_name', shape=(audios_num,), dtype='S80')
            hf.create_dataset(name='logmel', shape=(audios_num, audio_samples/overlap - 1, mel_bins), dtype=np.float32)
            hf.create_dataset(name='label', shape=(audios_num,), dtype=np.int16)
            for i, (audio_name, label) in enumerate(list(zip(audio_names, covid_statuses))):
                hf['audio_name'][i] = audio_name.encode()
                hf['label'][i] = label
                audio_path = os.path.join(audio_dir, audio_name + '.wav')
                (audio, fs) = read_audio(audio_path, max_len=audio_samples, target_fs=sample_rate)
                soundfile.write(file='temp.wav', data=audio, samplerate=fs)
                feature = calculate_logmel(audio_path='temp.wav',
                                           sample_rate=sample_rate,
                                           feature_extractor=feature_extractor)
                hf['logmel'][i] = feature
                os.remove('temp.wav')
                #time.sleep(1)
            print('There are total {} segments in the coughvid hdf5 file'.format(i))

        elif dataset == 'dicova':
            hf.create_dataset(name='audio_name', shape=(audios_num,), dtype='S80')
            hf.create_dataset(name='logmel', shape=(audios_num, audio_samples/overlap - 1, mel_bins), dtype=np.float32)
            hf.create_dataset(name='label', shape=(audios_num,), dtype=np.int16)
            for i, (audio_name, label) in enumerate(list(zip(audio_names, covid_statuses))):
                hf['audio_name'][i] = audio_name.encode()
                hf['label'][i] = label
                audio_path = os.path.join(audio_dir, audio_name + '.flac')
                (audio, fs) = read_audio(audio_path, max_len=audio_samples, target_fs=sample_rate)
                soundfile.write(file='temp.wav', data=audio, samplerate=fs)
                feature = calculate_logmel(audio_path='temp.wav',
                                           sample_rate=sample_rate,
                                           feature_extractor=feature_extractor)
                hf['logmel'][i] = feature
                os.remove('temp.wav')
                #time.sleep(1)
            print('There are total {} segments in the dicova hdf5 file'.format(i))


  
    print('Save to hdf5 located at {}'.format(hdf5_path))
    print('Time spent: {} s'.format(time.time() - since_time))


if __name__ == '__main__':

    # this part is for debugging
    DATASET_DIR = '../workspace'
    WORKSPACE = '../workspace'
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='logmel')
    parser.add_argument('--dataset_dir', type=str, default=DATASET_DIR)
    parser.add_argument('--workspace', type=str, default=WORKSPACE)
    parser.add_argument('--dataset', type=str, choices=['flusense', 'coughvid', 'dicova', 'compare'])
    args = parser.parse_args()
    if args.mode == 'logmel':
        calculate_features(args)
    else:
        raise Exception('Incorrect arguments!')
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='logmel')
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--workspace', type=str, required=True)
    args = parser.parse_args()
    if args.mode == 'logmel':
        calculate_features(args)
    else:
        raise Exception('Incorrect arguments!')
    '''
