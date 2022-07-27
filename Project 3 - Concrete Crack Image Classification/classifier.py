import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, losses
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os, datetime
import matplotlib.pyplot as plt

#%% load data
SEED = 82258
IMG_SIZE = (100, 100)
BATCH = 32

train_ds = keras.utils.image_dataset_from_directory(
    r"C:\Training\ML\AI07_training_projects_datasets\concrete",
    validation_split=0.2,
    subset="training",
    seed=SEED,
    batch_size=BATCH)

val_ds = keras.utils.image_dataset_from_directory(
    r"C:\Training\ML\AI07_training_projects_datasets\concrete",
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    batch_size=BATCH)

CLASS_NAMES = train_ds.class_names

#%% prefetch
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

#%% create NN
keras.backend.clear_session()

conv_layers = 2
clsf_layers = 3
layer_activation = 'relu'

model = keras.Sequential()

# normalization layer
model.add(layers.Rescaling(1./255))

# augmentation
model.add(layers.RandomFlip("horizontal_and_vertical"))
model.add(layers.RandomRotation(0.2))

# convolutional layer
for i in range(conv_layers):
    model.add(layers.Conv2D(2**(i+3), (3, 3), padding='same', 
                            activation=layer_activation))
    model.add(layers.Conv2D(2**(i+3), (3, 3), padding='same', 
                            activation=layer_activation))
    model.add(layers.MaxPool2D((3, 3)))
model.add(layers.Flatten())

# classification layer
for i in range(clsf_layers):
    model.add(layers.Dense(2**(clsf_layers-i+3), 
                           activation=layer_activation))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(len(CLASS_NAMES)))

#%% callbacks
es = EarlyStopping(patience=5, restore_best_weights=True)

log_dir = "log/project_3/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tb = TensorBoard(log_dir=os.path.join(
    r'C:\Training\ML\AI07_training_projects_datasets', log_dir))

#%% compile 
EPOCHS = 20
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_ds,
                    validation_data=val_ds,
                    batch_size=BATCH, epochs=EPOCHS, verbose=1,
                    callbacks=[es, tb])


#%% visualize
plot_model(model, show_shapes=True, show_layer_activations=True)
epochs_x_axis = history.epoch

plt.plot(epochs_x_axis, history.history['accuracy'],
         label='Training accuracy')
plt.plot(epochs_x_axis, history.history['val_accuracy'],
         label='Validation accuracy')
plt.title('Training vs Validation Accuracy')
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
test_result = model.evaluate(val_ds)

#%% deploy
image_batch, label_batch = val_ds.as_numpy_iterator().next()
predictions = np.argmax(model.predict(image_batch), axis=1)

plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(image_batch[i].astype('int'))

    if label_batch[i] != predictions[i]:
        plt.title(f'not a {CLASS_NAMES[predictions[i]]}?')
    else:
        plt.title(f'{CLASS_NAMES[predictions[i]]}')
    x = image_batch[i]
    plt.axis('off')
plt.savefig('example.png')