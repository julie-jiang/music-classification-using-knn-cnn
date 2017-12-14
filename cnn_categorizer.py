from __future__ import print_function
from __future__ import division
import argparse
import glob
import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support
from keras.models import Sequential
from keras.optimizers import Adam, SGD
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.initializers import TruncatedNormal, Constant, RandomNormal

classes = {"string": 0, "keyboard": 1, "vocal": 2, "guitar": 3, "brass": 4}
n_classes = len(classes)
save_dir   = "/data/juliej/results/"

def compile_model(input_shape):
    
    n_hidden_units_one = 56
    lr = 0.0001

    model = Sequential()
    model.add(Conv2D(n_hidden_units_one, kernel_size=(3, 3), 
                     input_shape=input_shape, activation='relu', 
                     kernel_initializer=RandomNormal(stddev=0.1),
                     bias_initializer=RandomNormal(stddev=0.1)))
    model.add(MaxPooling2D())
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(n_classes, activation='softmax',
                    kernel_initializer=RandomNormal(stddev=0.1),
                    bias_initializer=RandomNormal(stddev=0.1)))
    adam = Adam(lr=lr)

    model.compile(optimizer=adam, 
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    print(model.summary())

    return model


def train(tr_features, tr_labels, ts_features, ts_labels):
    # Hyperparamters
    input_shape = tr_features.shape[1:]
    model = compile_model(input_shape)
    epochs = 2000
    batch_size = 32
    patience = 20
    
    # Callbacks
    earlyStopping = EarlyStopping(monitor='val_acc', patience=patience, verbose=1)
    weights_file = os.path.join(save_dir, 'weights.{epoch:02d}-{val_acc:.2f}.hdf5')
    modelCheckpt = ModelCheckpoint(weights_file, monitor='val_acc', 
                                   verbose=1, save_best_only=True,
                                   save_weights_only=True)

    tsboard = TensorBoard(log_dir=os.path.join(save_dir, 'logs'), 
                          batch_size=batch_size)

    # Train
    model.fit(tr_features, tr_labels, 
              batch_size=batch_size, 
              epochs=epochs, 
              verbose=1,
              validation_data=(ts_features, ts_labels), 
              shuffle=True, 
              callbacks=[earlyStopping, modelCheckpt, tsboard])


def test(ts_features, ts_labels, weights):
    input_shape = ts_features.shape[1:]
    model = compile_model(input_shape)
    model.load_weights(weights)
    scalars = model.evaluate(ts_features, ts_labels)
    print("\nEvaluation loss: %f, accuracy %f" % (scalars[0], scalars[1]))
    predictions = model.predict(ts_features)
    ts_labels = np.argmax(ts_labels, axis=1)
    predictions =np.argmax(predictions, axis=1)
    con_mat = tf.confusion_matrix(ts_labels, predictions, num_classes=n_classes)
    with tf.Session():
        print("Confusion matrix: \n", tf.Tensor.eval(con_mat))
    p, r, f, _ = precision_recall_fscore_support(ts_labels, predictions,)
    print("Precision:", p, "\nRecall:", r, "\nFscore:", f)



def load_data(datafile):
    data = np.load(datafile)
    return data['tr_features'], data['tr_labels'], \
           data['ts_features'], data['ts_labels']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('datafile', metavar='<datafile.npz>',
                        help='Dataset in the format of numpy arrays.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--train',
                        help='Train a CNN with the given numpy dataset')
    group.add_argument('--test',
                        dest='weights', metavar='<weights.hdf5>', 
                        help='Weights file for testing')

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    tr_features, tr_labels, ts_features, ts_labels = load_data(args.datafile)
    if args.train:
        print("Training CNN with data from", args.datafile)
        train(tr_features, tr_labels, ts_features, ts_labels)
    else:
        print("Testing CNN with data from", args.datafile, "and weights", args.weights)
        test(ts_features, ts_labels, args.weights)

    












