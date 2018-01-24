# 认狗程序：keras预训练的ResNet50，当然不光认狗
# 认不出东亚犬种

import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from os import listdir, makedirs
from os.path import join, exists, expanduser
from tqdm import tqdm
from sklearn.metrics import log_loss, accuracy_score
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications import xception
from keras.applications import inception_v3
from keras.applications.vgg16 import preprocess_input, decode_predictions
from sklearn.linear_model import LogisticRegression
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.preprocessing.image import load_img, img_to_array
from keras.layers import Input
from keras.utils.generic_utils import Progbar
from keras.preprocessing.image import ImageDataGenerator
import keras, cv2, os, numpy as np, pandas as pd, time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


model = ResNet50(weights='imagenet')
aaa=load_img('/home/m/桌面/3.jpeg', target_size=(224, 224))
img = np.array([img_to_array(aaa)]) #/ 255
print(img.shape)
bbb=model.predict(img)
_, imagenet_class_name, prob = decode_predictions(bbb, top=1)[0][0]
imagenet_class_name



######## 上下效果一样，下面是比较像来源程序的版本，上面是更自我的版本	######

model = ResNet50(weights='imagenet')
aaa=load_img('/home/m/桌面/4.jpeg', target_size=(224, 224))
aaa=img_to_array(aaa)
x = preprocess_input(np.expand_dims(aaa.copy(), axis=0)) #/ 255
#print(img.shape)
bbb=model.predict(x)
_, imagenet_class_name, prob = decode_predictions(bbb, top=1)[0][0]
imagenet_class_name





