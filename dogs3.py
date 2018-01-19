

from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.preprocessing.image import load_img, img_to_array
from keras.layers import Input

import keras, cv2, os, numpy as np, pandas as pd, time
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


x_train = []
y_train = []



XY=16

path_in_labels = '/home/m/桌面/dogs/data/labels.csv'
labels = pd.read_csv(path_in_labels).set_index('id')['breed']

path_in_train = '/home/m/桌面/dogs/data/train/'
list_1 = os.listdir(path_in_train)
for img_name in tqdm(list_1):
	img = load_img(path_in_train + img_name, target_size=(XY,XY))
	#img = img_to_array(img).flatten() / 255
	img = img_to_array(img) / 255
	img = [img.tolist()]
	x_train += img
	y_train += [labels[img_name[:-4]]]


###############



classes = len(pd.Series(y_train).unique())		# 几种车
	
x_train, x_vali, y_train, y_vali = train_test_split(np.array(x_train), np.array(y_train))
#del x_all, y_all



encoder = LabelEncoder()
hotencoder = OneHotEncoder(sparse=False)
y_train = hotencoder.fit_transform(pd.DataFrame({'a':encoder.fit_transform(y_train)}))
y_vali = hotencoder.transform(pd.DataFrame({'a':encoder.transform(y_vali)}))


model = Sequential()
#model.add(Dense(input_dim=100*100,units=633,activation='relu'))
#model.add(Conv2D(filters= 32, kernel_size=(5,5), padding='Same', activation='relu',input_shape=(28,28,1)))

model.add(Conv2D(XY, (3, 3), activation='relu', padding='same', name='block1_conv1', input_shape=x_train.shape[1:]))	#(100,100,3), data_format='channels_last'))		#input_dim=100*100)),
model.add(Conv2D(XY, (3, 3), activation='relu', padding='same', name='block1_conv2'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))
'''
model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))
'''
model.add(Flatten(name='flatten'))
model.add(Dense(4096, activation='relu', name='fc1'))
model.add(Dense(4096, activation='relu', name='fc2'))
model.add(Dense(classes, activation='softmax', name='predictions'))

#model.compile(loss='mse', optimizer=SGD(lr=0.1), metrics=['accuracy'])	

opt = keras.optimizers.rmsprop(lr=0.0001)	#, decay=1e-6)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=50, epochs=50)

result = model.evaluate(x_vali, y_vali)
print(result)


###############

x_test = []

path_in_test = '/home/m/桌面/dogs/data/test/'
list_2 = os.listdir(path_in_test)
for img_name in tqdm(list_2):
	img = load_img(path_in_test + img_name, target_size=(XY,XY))
	#img = img_to_array(img).flatten() / 255
	img = img_to_array(img) / 255
	img = [img.tolist()]
	x_test += img
	

y_test = model.predict(x_test)







###########
path_in = '/home/m/桌面/cars/'
list_1 = ['jili','baoma','mumaren']
for car_name in list_1:
	list_2 = os.listdir(path_in + car_name)
	for img_name in tqdm(list_2[400:500]):
		img = load_img(path_in + car_name + '/' + img_name, target_size=(70,70))
		#img = img_to_array(img).flatten() / 255
		img = img_to_array(img) / 255
		img = [img.tolist()]
		x_all += img
		y_all += [car_name]

x_vali = np.array(x_all)
y_vali = np.array(y_all)

y_vali = hotencoder.transform(pd.DataFrame({'a':encoder.transform(y_vali)}))

y_pred = model.predict(x_vali)
y_pred[y_pred<1.0] = 0

accu = 1 - (len(y_pred)*classes - sum((y_pred == y_vali).flatten()))/(2*len(y_pred))










