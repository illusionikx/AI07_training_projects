import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, losses, applications
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os, datetime
import matplotlib.pyplot as plt


#%% import datasets
ds = pd.read_csv(r"C:\Training\ML\AI07_training_projects_datasets\heart.csv",
                 header=0,
                 index_col=False)
features = ds.copy()
labels = features.pop('target')


#%% split
SEED = 76153
N_INPUT = features.shape[1]


x_train, x_test, y_train, y_test = train_test_split(features, labels,
                                                    test_size=0.3,
                                                    random_state=SEED)

#%% build model

keras.backend.clear_session()

normalizer = layers.Normalization(axis=-1)
normalizer.adapt(x_train)

model = keras.Sequential()
model.add(normalizer)

n_layers = 4

for i in range(n_layers):
    model.add(layers.Dense(2**(n_layers-i+4), 
                           activation='selu'))
    model.add(layers.Dropout(0.2))
model.add(layers.Dense(1))

model.summary()

#%% callbacks
es = EarlyStopping(patience=20, restore_best_weights=True)

log_dir = "log/project_1/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tb = TensorBoard(log_dir=os.path.join(
    r'C:\Training\ML\AI07_training_projects_datasets', log_dir))

#%% compile 
BATCH_SIZE = features.shape[0]
EPOCHS = 300

model.compile(optimizer='adam',
              loss=losses.BinaryCrossentropy(from_logits=True), 
              metrics=['binary_accuracy'])
history = model.fit(x_train, y_train,
                    validation_data=(x_test, y_test),
                    batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1,
                    callbacks=[es, tb])

#%% visualize
plot_model(model, to_file='model.png', rankdir='LR')
epochs_x_axis = history.epoch

plt.plot(epochs_x_axis, history.history['binary_accuracy'],
         label='Training accuracy')
plt.plot(epochs_x_axis, history.history['val_binary_accuracy'], label='Validation accuracy')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.figure()
plt.show()

plt.plot(epochs_x_axis, history.history['loss'],
         label='Training loss')
plt.plot(epochs_x_axis, history.history['val_loss'],
         label='Validation loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.figure()
plt.show()