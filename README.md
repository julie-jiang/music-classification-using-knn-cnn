# music-classification-using-knn-cnn
## Report and Presentation
For detailed project report and write up, please check [this file](music-classification-using-knn-cnn-report.pdf).

For a abstract overview of our project, please refer to our [slides](https://docs.google.com/presentation/d/1Zt8KGg0PVwtuc5_7TKHfrMUbzMaW1T4H1juTLiXYQNw/edit?usp=sharing).

## Dataset
The Nsynth Dataset we used can be obtained [here](https://magenta.tensorflow.org/datasets/nsynth). We processed a subset of this dataset for use in our training. Due to the large size of the file (> 1GB), we could not put it up here. But we've included few samples of files in [eg/tr](eg/tr) and [eg/ts](eg/ts).

## CNN Usage

Prepare two directory containing your training set and validation set, respectively. Your dataset which should be a set of `*.wav` files.

You might want to change these variables at the top of `featExtractionCNN.py` to reflect your dataset.
```
# Unique index for every class/label
classes = {"string": 0, "keyboard": 1, "vocal": 2, "guitar": 3, "brass": 4}
train_dir = "nsynth-train" # Directory to your training set 
valid_dir = "nsynth-valid" # Directory to your test set
datafile = "./dataCNN.npz" # What to save the numpy arrays as
```
Now run the following to produce a `.npz` file that contains the processed data. 
```
python3 featExtractionCNN.py
```

With this file, you can proceed to test your dataset with a set of pretrained weights.
```
python3 --test eg/weightsCNN.hdf5 [data.npz]
```
To train a new model, run
```
python3 --train [data.npz]
```

## KNN Usage

Generate a `.npz` from the dataset, as described above.  Then, run the kNN.
```
python knn_categorizer.py
```

## Dependencies
The CNN model works with python 2 or 3.
The KNN model works with python 2
- [Keras](https://keras.io/) `2.0.6` (For CNN only)
- [Tensorflow](https://www.tensorflow.org/) `1.4.0` or `1.2.1` (For CNN only)
- [Librosa](http://librosa.github.io/) `0.5.1`
- Sklearn `0.18.1`
- Numpy `1.13.3`
- hdf5
