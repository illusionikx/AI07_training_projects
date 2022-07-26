from tensorflow import keras
from tensorflow.keras import layers, optimizers, losses
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd
import os, datetime
import matplotlib.pyplot as plt

#%% import datasets
df = pd.read_csv(r"C:\Training\ML\AI07_training_projects_datasets\garments_worker_productivity.csv",
                 header=0,
                 index_col=False)

#%% pre-processing
# dropping date because irrelevant
df.drop(columns='date', inplace=True)
# replacing missing wip with 0
df['wip'].fillna(0, inplace=True)
# correct errors in department column
df['department'] = df['department'].replace('sweing', 'sewing')
df['department'] = df['department'].replace('finishing ', 'finishing')
# onehot encode quarter, department, day
df = pd.get_dummies(df, columns=['quarter', 'department', 'day']) 

#%% split model
features = df.copy()
labels = features.pop('actual_productivity')

SEED = 14800
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

n_layers = 5

for i in range(n_layers):
    model.add(layers.Dense(2**(n_layers-i+4), 
                           activation='selu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(1))

model.summary()

#%% callbacks
es = EarlyStopping(patience=20, restore_best_weights=True)

log_dir = "log/project_2/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tb = TensorBoard(log_dir=os.path.join(
    r'C:\Training\ML\AI07_training_projects_datasets', log_dir))

#%% compile 
BATCH_SIZE = features.shape[0]
EPOCHS = 300

model.compile(optimizer='adam',
              loss='mae', 
              metrics=['RootMeanSquaredError'])
history = model.fit(x_train, y_train,
                    validation_data=(x_test, y_test),
                    batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1,
                    callbacks=[es, tb])

#%% visualize
plot_model(model, show_shapes=True, show_layer_activations=True)
epochs_x_axis = history.epoch

plt.plot(epochs_x_axis, history.history['root_mean_squared_error'],
         label='Training Root Mean Squared Error')
plt.plot(epochs_x_axis, history.history['val_root_mean_squared_error'],
         label='Validation Root Mean Squared Error')
plt.title('Training vs Validation Root Mean Squared Error')
plt.savefig('metrics.png')
plt.xlabel('Epoch')
plt.ylabel('')
plt.legend()
plt.figure()
plt.show()


plt.plot(epochs_x_axis, history.history['loss'],
         label='Training loss')
plt.plot(epochs_x_axis, history.history['val_loss'],
         label='Validation loss')
plt.title('Training vs Validation Loss')
plt.savefig('loss.png')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.figure()
plt.show()


#%% predict and show report
test_result = model.evaluate(x_test, y_test)

