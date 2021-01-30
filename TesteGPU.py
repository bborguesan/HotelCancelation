# Imports

# Imports para manipulação e visualização de dados
import numpy as np
import pandas as pd
import matplotlib as m
import matplotlib.pyplot as plt

# Imports para pré-processamento e avaliação
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

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
y = to_categorical(y, num_classes = None)
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, shuffle = True, test_size = 0.2, random_state = 420)
scaler = MinMaxScaler()
X_treino = scaler.fit_transform(X_treino)
X_teste = scaler.transform(X_teste)

# Criamos o modelo
modelo = Sequential()
modelo.add(Dense(200, input_dim = X.shape[1], activation = 'relu'))
modelo.add(Dropout(0.1))
modelo.add(Dense(200, activation = 'relu'))
modelo.add(Dropout(0.2))    
modelo.add(Dense(200, activation = 'relu'))
modelo.add(Dropout(0.2))
modelo.add(Dense(100, activation = 'relu'))
modelo.add(Dropout(0.1))
modelo.add(Dense(2, activation = 'softmax'))

# Usaremos como otimizador o algoritmo Nadam pois o conjunto de dados é complexo
# https://keras.io/api/optimizers/Nadam/
otimizador = Nadam(learning_rate = 0.0001, 
                   beta_1 = 0.9, 
                   beta_2 = 0.999, 
                   epsilon = 1e-07)

# Compilamos o modelo
modelo.compile(optimizer = otimizador, loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Vamos criar 2 callbacks
# Um para finalizar o treinamento se depois de 20 passadas o erro não mudar
# Outro para reduzir a taxa de aprendizagem se o erro não mudar após 5 passadas
callbacks = [EarlyStopping(monitor = "loss", patience = 20), 
             ReduceLROnPlateau(monitor = "loss", patience = 5)]

# Hiperparâmetros
num_epochs = 65
batch_size = 32

# Treinamento

print("\nTreinamento Iniciado.\n")

history = modelo.fit(X_treino, y_treino, epochs = num_epochs, batch_size = batch_size, callbacks = callbacks)

print("\nTreinamento Concluído.\n")

# Plot da Acurácia em Treino
plt.figure(figsize = [10,8])
plt.title("Curva de Aprendizado do Modelo - Acurácia")
plt.plot(history.history['accuracy'], label = 'Acurácia em Treino')
plt.xlabel("Epochs")
plt.legend()
plt.grid()
plt.savefig('Acuracia-fitOriginal.png')

# Plot do Erro em Treino
plt.figure(figsize = [10,8])
plt.title("Curva de Aprendizado do Modelo - Erro")
plt.plot(history.history['loss'], label = 'Erro em Treino')
plt.xlabel("Epochs")
plt.legend()
plt.grid()
plt.savefig('Erro-fitOriginal.png')


# Fazemos as previsões com os dados de teste
previsoes = modelo.predict(X_teste)
# Convertemos as previsões em previsões de classe
y_pred = (previsoes > 0.5) 
# Calculamos a acurácia comparando valor real com valor previsto
print(accuracy_score(y_teste, y_pred))

# Salva o modelo
modelo.save("modelo/modelo-fitOriginal.h5")