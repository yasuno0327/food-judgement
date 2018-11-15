import tensorflow as tf
import keras
from keras.models import Sequential, Model
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, Input, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.applications.vgg16 import VGG16
from keras.utils import np_utils

from PIL import Image
import numpy as np
import re
import os

config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 56} )
sess = tf.Session(config=config)
keras.backend.set_session(sess)

train_images = [] # トレーニングデータ格納用配列
train_labels = [] # トレーニングラベル格納用配列
test_images = [] # テストデータ格納用配列
test_labels = [] # テストデータ格納用配列

for line in open(os.getcwd() + "/meta/train.txt"):
  # Store label
  regex = '[^0-9/]*'
  label = re.match(regex, line).group(0)
  train_labels.append(label)
  # Resize image and store
  filename = line.strip() + ".jpg"
  image_path = os.getcwd() + "/data/" + filename
  image = np.array(Image.open(image_path).convert("L").resize((140,140)))
  image = image.reshape(1, 19600).astype("float32")[0]
  train_images.append(image / 255.)

for line in open(os.getcwd() + "/meta/test.txt"):
  # Store test label
  regex = '[^0-9/]*'
  label = re.match(regex, line).group(0)
  test_labels.append(label)
  # Resize test image and store
  filename = line.strip() + ".jpg"
  image_path = os.getcwd() + "/data/" + filename
  image = np.array(Image.open(image_path).convert("L").resize((140,140)))
  image = image.reshape(1, 19600).astype("float32")[0]
  test_images.append(image / 255.)

train_images = np.array(train_images)
train_labels = np.array(train_labels)
train_labels = np_utils.to_categorical(train_labels, 101)

test_images = np.array(test_images)
test_labels = np.array(test_labels)
test_labels = np_utils.to_categorical(test_labels, 101)

input_tensor = Input(shape=(140,140,3))

# Define model

vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)

vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)

top_model = Sequential()

top_model.add(Conv2D(filters=32, kernel_size=(28,28)))
top_model.add(Activation("relu"))
top_model.add(Conv2D(filters=32, kernel_size=(28,28)))
top_model.add(Activation("relu"))
top_model.add(MaxPooling2D(pool_size=(2,2)))
top_model.add(Dropout(0.3))

top_model.add(Conv2D(64, (28, 28), padding="same"))
top_model.add(Activation("relu"))
top_model.add(Conv2D(64, (28,28)))
top_model.add(Activation("relu"))
top_model.add(MaxPooling2D(pool_size=(2,2)))
top_model.add(Dropout(0.3))

top_model.add(Flatten())
top_model.add(Dense(512))
top_model.add(Activation("relu"))
top_model.add(Dropout(0.5))
top_model.add(Dense(101))
top_model.add(Activation("softmax"))

model = Model(inputs=vgg16.input, outputs=top_model(vgg16.output))

for layer in model.layers[:19]:
    layer.trainable = False

google_drive_dir = 'gdrive/My Drive'
base_file_name = 'test1'

# Google Driveに逐次モデルのスナップショットを保存
checkpointer = ModelCheckpoint(filepath = google_drive_dir + base_file_name + '.{epoch:02d}-{val_loss:.2f}.hdf5', verbose=1, save_best_only=True, monitor='val_acc', mode='max')

model.fit(train_images, train_labels, batch_size=128, epochs=3)

model.compile(
    optimizer=tf.train.RMSPropOptimizer(learning_rate=0.01),
    loss='categorical_crossentropy',
    metrics=['categorical_accuracy']
)

import json
json_model = model.to_json()
with open('data.json', 'w') as outfile:
    json.dump(json_model, outfile)

scores = model.evaluate(test_images, test_labels, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
