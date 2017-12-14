from __future__ import print_function
from __future__ import division
import glob
import os
import numpy as np
import librosa
import cfg

datafile = "data40.npz"

def extract_feature(file_name):
    X, sr = librosa.load(file_name)
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sr, n_mfcc=cfg.n_mfcc).T, axis=0)
    return mfccs

def parse_audio_files(parent_dir,sub_dir,file_ext="*.wav"):
    print("parsing audio from ", sub_dir)
    features, labels = np.empty((0, cfg.n_mfcc)), np.empty(0)
    for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
        mfccs = extract_feature(fn)
        features = np.vstack([features, mfccs])
        labels = np.append(labels, cfg.classes[fn.split('/')[-1].split('_')[0]])
    return np.array(features), one_hot_encode(np.array(labels, dtype=np.int))

def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels,n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode



def load_data():
    data = np.load(os.path.join(cfg.parent_dir, datafile))
    return data['tr_features'], data['tr_labels'], \
           data['ts_features'], data['ts_labels']


def process_raw_data():
    print("about to starting parsing audio")
    tr_features, tr_labels = parse_audio_files(cfg.parent_dir, cfg.tr_sub_dir)
    ts_features, ts_labels = parse_audio_files(cfg.parent_dir, cfg.ts_sub_dir)

    # data = np.load(os.path.join(parent_dir, "data1.npz"))

    np.savez(os.path.join(cfg.parent_dir, datafile), 
             tr_features=tr_features, tr_labels=tr_labels,
             # tr_features=data['tr_features'], tr_labels=data['tr_labels'],
             ts_features=ts_features, ts_labels=ts_labels)

    print("processed data saveed to", os.path.join(cfg.parent_dir, datafile))

    return tr_features, tr_labels, ts_features, ts_labels
