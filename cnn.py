import keras
from keras.models import Sequential
from keras.utils import np_utils

from PIL import Image
import numpy as np
import re
import os

train_images = [] # トレーニングデータ格納用配列
train_labels = [] # トレーニングラベル格納用配列

for line in open(os.getcwd() + "/meta/train.txt"):
  train_labels.append(line)
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

