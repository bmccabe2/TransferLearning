import tensorflow as tf
from tensorflow.python.keras.applications import vgg16
from tensorflow.python.keras.optimizers import SGD
from tensorflow.keras import datasets
from tensorflow.python.keras.utils import multi_gpu_model
import numpy as np

# CIFAR10 training and test data sets.
NUM_CLASSES = 10
IMG_SIZE = 32
PER_CORE_BATCH_SIZE = 40
BATCH_SIZE = 512


print(tf.__version__)

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0


label_names, counts = np.unique(train_labels, return_counts=True)

IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
VGG16_MODEL = vgg16.VGG16(input_shape=IMG_SHAPE,
            include_top=False,
                        weights='imagenet')
VGG16_MODEL.summary()

VGG16_MODEL.trainable=False
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = tf.keras.layers.Dense(len(label_names),activation='softmax')
  
model = tf.keras.Sequential([
  VGG16_MODEL,
  global_average_layer,
  prediction_layer
])

model = multi_gpu_model(model=model, gpus=2)

model.compile(optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=0.01), 
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=["accuracy"])

print(train_images.shape)

history = model.fit(train_images, train_labels,
                    epochs=100, 
                    batch_size=BATCH_SIZE,
                    validation_data=(test_images, test_labels),
                    shuffle=True, verbose=2)

# history = model.fit(train_images, train_labels,
#                     epochs=100, 
#                     steps_per_epoch=150,
#                     validation_steps=312,
#                     validation_data=(test_images, test_labels),
#                     verbose=1)

prediction = model.evaluate(test_images, test_labels, verbose=2, steps=100)

print (prediction)