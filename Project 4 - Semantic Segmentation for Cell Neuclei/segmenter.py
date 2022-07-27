import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses, optimizers, callbacks
from tensorflow.keras.utils import plot_model
from tensorflow_examples.models.pix2pix import pix2pix
from IPython.display import clear_output
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os

#%% import images
root_path = r"C:\Training\ML\AI07_training_projects_datasets\nuclei"

def image_converter(path, grayscale=False):
    images = []
    for image_file in os.listdir(path):
        if grayscale:
            img = cv2.imread(os.path.join(path, image_file), cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imread(os.path.join(path, image_file))
            img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 128))
        images.append(img)
    images = np.array(images)
    
    if grayscale:
        images = np.expand_dims(images, axis=-1)
        images = np.round(images/255)
    else:
        images = images / 255.0
    
    return tf.data.Dataset.from_tensor_slices(images)

x_train = image_converter(os.path.join(root_path, 'train', 'inputs'))
y_train = image_converter(os.path.join(root_path, 'train', 'masks'), True)
x_test = image_converter(os.path.join(root_path, 'test', 'inputs'))
y_test = image_converter(os.path.join(root_path, 'test', 'masks'), True)

train_ds = tf.data.Dataset.zip((x_train, y_train))
test_ds = tf.data.Dataset.zip((x_test, y_test))

#%% Create a subclass layer for data augmentation
class Augment(layers.Layer):
    def __init__(self,seed=42):
        super().__init__()
        self.augment_inputs = layers.RandomFlip(mode='horizontal',seed=seed)
        self.augment_labels = layers.RandomFlip(mode='horizontal',seed=seed)
        
    def call(self,inputs,labels):
        inputs = self.augment_inputs(inputs)
        labels = self.augment_labels(labels)
        return inputs,labels
    
#%% Build the input dataset
keras.backend.clear_session()

BUFFER_SIZE = 1000
TRAIN_SIZE = len(train_ds)
BATCH_SIZE = TRAIN_SIZE//10
STEPS_PER_EPOCH = TRAIN_SIZE//BATCH_SIZE

train_batches = (
    train_ds
    .cache()
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE)
    .repeat()
    .map(Augment())
    .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

test_batches = test_ds.batch(BATCH_SIZE)

#%% visualize examples
def display(display_list):
    plt.figure(figsize=(15,15))
    title = ['Input Image','True Mask','Predicted Mask']
    
    for i in range(len(display_list)):
        plt.subplot(1,len(display_list),i+1)
        plt.title(title[i])
        plt.imshow(keras.utils.array_to_img(display_list[i]))
        plt.axis('off')
        
    plt.show()

for images, masks in train_batches.take(2):
    sample_image,sample_mask = images[0],masks[0]
    display([sample_image,sample_mask])
    
#%% adding sample weight
def add_sample_weights(image, label):
  # The weights for each class, with the constraint that:
  #     sum(class_weights) == 1.0
  class_weights = tf.constant([2.0, 2.0, 1.0])
  class_weights = class_weights/tf.reduce_sum(class_weights)

  # Create an image of `sample_weights` by using the label at each pixel as an 
  # index into the `class weights` .
  sample_weights = tf.gather(class_weights, indices=tf.cast(label, tf.int32))

  return image, label, sample_weights
    
#%% create model
keras.backend.clear_session()

base_model = keras.applications.MobileNetV2(input_shape=[128,128,3],include_top=False)

#8.2. List down some activation layers
layer_names = [
    'block_1_expand_relu',   # 64x64
    'block_3_expand_relu',   # 32x32
    'block_6_expand_relu',   # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',      # 4x4
    ]

base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

#Define the feature extraction model
down_stack = keras.Model(inputs=base_model.input, outputs=base_model_outputs)
down_stack.trainable = False

#Define the upsampling path
up_stack = [
    pix2pix.upsample(512, 3),  # 4x4 -> 8x8
    pix2pix.upsample(256, 3),  # 8x8 -> 16x16
    pix2pix.upsample(128, 3),  # 16x16 -> 32x32
    pix2pix.upsample(64, 3),   # 32x32 -> 64x64
]

def unet_model(output_channels:int):
    inputs = layers.Input(shape=[128,128,3])
    #Apply functional API to construct U-Net
    #Downsampling through the model
    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])
    
    #Upsampling and establishing the skip connections(concatenation)
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = layers.Concatenate()
        x = concat([x,skip])
        
    #This is the last layer of the model (output layer)
    last = layers.Conv2DTranspose(
        filters=output_channels,kernel_size=3,strides=2,padding='same') #64x64 --> 128x128
    
    x = last(x)
    
    return keras.Model(inputs=inputs, outputs=x)

#%% construct u-net
OUTPUT_CLASSES = 2

model = unet_model(output_channels=OUTPUT_CLASSES)
#Compile the model
model.compile(optimizer='adam',
              loss=losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#%% show predictions
def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask,axis=-1)
    pred_mask = pred_mask[...,tf.newaxis]
    return pred_mask[0]

def show_predictions(dataset=None, num=1):
    if dataset:
        for image,mask in dataset.take(num):
            pred_mask = model.predict(image)
            display([image[0],mask[0],create_mask(pred_mask)])
            
    else:
        display([sample_image,sample_mask,
                 create_mask(model.predict(sample_image[tf.newaxis,...]))])

#%% create callback to show training
class DisplayCallback(callbacks.Callback):
    def on_epoch_end(self,epoch,logs=None):
        clear_output(wait=True)
        show_predictions()
        print('\nSample Prediction after epoch {}\n'.format(epoch+1))
        
#%% train model
EPOCHS = 50

history = model.fit(train_batches.map(add_sample_weights),
                    validation_data=test_batches,
                    epochs=EPOCHS,
                    steps_per_epoch=STEPS_PER_EPOCH,
                    callbacks=[DisplayCallback()])

#%% plot result
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
plt.ylim([0, 1])
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
plt.ylim([0, 1])
plt.legend()
plt.figure()
plt.show()

#%% deploy model
show_predictions(test_batches,3)