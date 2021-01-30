import numpy as np
import pandas as pd
import matplotlib as m
import matplotlib.pyplot as plt

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')

import keras
from keras.models import Sequential
from keras.models import load_model 
from keras.layers import Dense, Input, Dropout
from keras.losses import categorical_crossentropy, binary_crossentropy
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Nadam

dataset = pd.read_pickle('datasetClean.pkl')

X = dataset.drop(["IsCanceled"], axis = 1)
y = dataset["IsCanceled"]
y = to_categorical(y, num_classes = None)
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, shuffle = True, test_size = 0.2, random_state = 420)
scaler = MinMaxScaler()
X_treino = scaler.fit_transform(X_treino)
X_teste = scaler.transform(X_teste)

modelo = load_model('modelo/modelo-fitOriginal.h5')

previsoes = modelo.predict(X_teste)
y_pred = (previsoes > 0.5) 
print(accuracy_score(y_teste, y_pred))
