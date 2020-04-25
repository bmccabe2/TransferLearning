import keras
import numpy as np
import cv2
import tensorflow as tf
from keras.applications import vgg16
from keras.optimizers import SGD
from keras import datasets
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras import optimizers
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer, UpSampling2D, BatchNormalization
import numpy as np


IMG_SIZE = 28
NUM_CLASSES = 10
BATCH_SIZE = 16

(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

### A Method of changin grayscale to rgb

dim = (32,32)
def to_rgb(img):
  img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA) 
  img_rgb = np.asarray(np.dstack((img, img, img)), dtype=np.uint8)
  return img_rgb

rgb_list = []
for i in range(len(train_images)):
  rgb = to_rgb(train_images[i])
  rgb_list.append(rgb)

rgb_arr = np.stack([rgb_list],axis=4)
train_images = np.squeeze(rgb_arr, axis=4)

rgb_list = []
for i in range(len(test_images)):
  rgb = to_rgb(test_images[i])
  rgb_list.append(rgb)

rgb_arr = np.stack([rgb_list],axis=4)
test_images = np.squeeze(rgb_arr, axis=4)
### END grayscale->rgb

train_images = train_images.astype('float32')
test_images = test_images.astype('float32')
train_images = train_images / 255.0
test_images = test_images / 255.0

train_labels = keras.utils.to_categorical(train_labels, NUM_CLASSES)
test_labels = keras.utils.to_categorical(test_labels, NUM_CLASSES)

print(train_images.shape)
print(test_images.shape)

label_names, counts = np.unique(train_labels, return_counts=True)

IMG_SHAPE = (32, 32, 3)

VGG16_MODEL = vgg16.VGG16(input_shape=IMG_SHAPE,
                        include_top=False,
                        weights='imagenet')
VGG16_MODEL.summary()

VGG16_MODEL.trainable=False

x = VGG16_MODEL.layers[-1].output
x = Flatten()(x)
x = BatchNormalization()(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.3)(x) # was .3
x = BatchNormalization()(x)
prediction = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(VGG16_MODEL.input, output=prediction)

for layer in VGG16_MODEL.layers:
    layer.trainable = False

opt = optimizers.RMSprop(lr=2e-5)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

print(test_labels.shape)

history = model.fit(train_images, train_labels,
                    epochs=100, 
                    batch_size=BATCH_SIZE,
                    validation_data=(test_images, test_labels),
                    verbose=2)

prediction = model.evaluate(test_images, test_labels, verbose=2, steps=100)

print (prediction)