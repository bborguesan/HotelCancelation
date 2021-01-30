# Imports

# Imports para manipulação e visualização de dados
import numpy as np
import pandas as pd
import matplotlib as m
import matplotlib.pyplot as plt

# Imports para pré-processamento e avaliação
import sklearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline


# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # Force to use CPU for test

# Imports para Deep Learning
import tensorflow as tf
# Lista o código de cada GPU

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs 'failed to create cublas handle: CUBLAS_STATUS_ALLOC_FAILED'
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


import keras
from keras.models import Sequential
from keras.layers import Dense, Input, Dropout
from keras.losses import categorical_crossentropy, binary_crossentropy
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Nadam

# Imports para formatação dos gráficos
from matplotlib.pylab import rcParams 
rcParams['axes.labelsize'] = 14
rcParams['xtick.labelsize'] = 12
rcParams['ytick.labelsize'] = 12
rcParams['text.color'] = 'k'
rcParams['figure.max_open_warning'] = 30
rcParams['figure.figsize'] = 10,8
m.style.use('ggplot')

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

dataset = pd.read_pickle('datasetClean.pkl')

X = dataset.drop(["IsCanceled"], axis = 1)
y = dataset["IsCanceled"]
# y = to_categorical(y, num_classes = None)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter = 250, verbose=1))
pipe.fit(X_train, y_train)
print(pipe.score(X_test, y_test))

# Salva o modelo
# modelo.save("modelo/modelo-fitNew.h5")