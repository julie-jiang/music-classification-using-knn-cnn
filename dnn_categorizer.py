import glob
import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support
from keras.models import Sequential
from keras.optimizers import Adam, SGD
from keras.layers import Dense, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.initializers import RandomNormal
from featExtraction import load_data, process_raw_data
import cfg

datafile = "data40.npz"

def train(tr_features, tr_labels, ts_features, ts_labels):
    # Hyperparamters
    epochs = 2000
    batch_size = 16
    n_dim = tr_features.shape[1]
    print(n_dim)
    n_classes = len(cfg.classes)
    n_hidden_units_one   = 30 * n_dim
    n_hidden_units_two   = 20 * n_dim
    n_hidden_units_three = 20 * n_dim 
    n_hidden_units_four  = n_dim 
    n_hidden_units_five  = n_dim // 2
    n_hidden_units_six   = n_dim // 4
    sd = 1 / np.sqrt(n_dim)
    lr = 0.0001
    patience = 20

    # Build and compile model

    model = Sequential()
    # model.add(Dropout(0.2))
    model.add(Dense(n_hidden_units_one, input_dim=n_dim, 
                    activation='sigmoid', 
                    kernel_initializer=RandomNormal(stddev=sd), 
                    bias_initializer=RandomNormal(stddev=sd)))
    model.add(Dropout(0.2))
    model.add(Dense(n_hidden_units_two, 
                    activation='sigmoid', 
                    kernel_initializer=RandomNormal(stddev=sd), 
                    bias_initializer=RandomNormal(stddev=sd)))
    model.add(Dropout(0.2))
    model.add(Dense(n_hidden_units_three, 
                    activation='sigmoid', 
                    kernel_initializer=RandomNormal(stddev=sd), 
                    bias_initializer=RandomNormal(stddev=sd)))
    model.add(Dropout(0.2))
    model.add(Dense(n_hidden_units_four, 
                    activation='sigmoid', 
                    kernel_initializer=RandomNormal(stddev=sd), 
                    bias_initializer=RandomNormal(stddev=sd)))
    model.add(Dropout(0.2))
    model.add(Dense(n_hidden_units_five, 
                    activation='sigmoid', 
                    kernel_initializer=RandomNormal(stddev=sd), 
                    bias_initializer=RandomNormal(stddev=sd)))
    model.add(Dropout(0.2))
    model.add(Dense(n_hidden_units_six, 
                    activation='sigmoid', 
                    kernel_initializer=RandomNormal(stddev=sd), 
                    bias_initializer=RandomNormal(stddev=sd)))
    model.add(Dropout(0.2))
    model.add(Dense(n_classes, 
                    activation='softmax', 
                    kernel_initializer=RandomNormal(stddev=sd), 
                    bias_initializer=RandomNormal(stddev=sd)))

    adam = Adam(lr=lr)
    # sgd = 

    model.compile(optimizer=adam, 
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    print(model.summary())

    # Callbacks

    earlyStopping = EarlyStopping(
                        monitor='val_acc', patience=patience, verbose=1)
    weights_file = os.path.join(
                        cfg.save_dir, 'weights.{epoch:02d}-{val_acc:.2f}.hdf5')
    modelCheckpt = ModelCheckpoint(weights_file, monitor='val_acc', 
                                   verbose=1, save_best_only=True,
                                   save_weights_only=True)

    tsboard = TensorBoard(log_dir=os.path.join(cfg.save_dir, 'logs'), batch_size=batch_size)

    # model.load_weights("/data/juliej/results/weights.34-0.81.hdf5")
    # Train
    model.fit(tr_features, tr_labels, 
              batch_size=batch_size, 
              epochs=epochs, 
              verbose=1,
              validation_data=(ts_features, ts_labels), 
              shuffle=True, 
              # initial_epoch=35,
              callbacks=[earlyStopping, modelCheckpt, tsboard])

    # p,r,f,s = precision_recall_fscore_support(y_true, y_pred, average="micro")

if __name__ == "__main__":

    load = True

    if load:
        tr_features, tr_labels, ts_features, ts_labels = load_data()
    else:
        tr_features, tr_labels, ts_features, ts_labels = process_raw_data()

    train(tr_features, tr_labels, ts_features, ts_labels)
    












