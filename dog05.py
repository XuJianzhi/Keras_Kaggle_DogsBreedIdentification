

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

'''
model = Sequential()
#model.add(Dense(input_dim=100*100,units=633,activation='relu'))
#model.add(Conv2D(filters= 32, kernel_size=(5,5), padding='Same', activation='relu',input_shape=(28,28,1)))

rate_dropout_conv = 0.6	#都0.5就过拟合，都0.6就欠拟合，哎！
rate_dropout_dens = 0.6

model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', input_shape=x_train.shape[1:]))	#(100,100,3), data_format='channels_last'))		#input_dim=100*100)),
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))
model.add(Dropout(rate_dropout_conv))

model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))
model.add(Dropout(rate_dropout_conv))

model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))
model.add(Dropout(rate_dropout_conv))

model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))
model.add(Dropout(rate_dropout_conv))

model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))

model.add(Flatten(name='flatten'))
model.add(Dense(4096, activation='relu', name='fc1'))
model.add(Dropout(rate_dropout_dens))
model.add(Dense(4096, activation='relu', name='fc2'))
model.add(Dropout(rate_dropout_dens))
model.add(Dense(classes, activation='softmax', name='predictions'))
'''

# 建立一个简单的卷积神经网络
def build_model():
    model = Sequential()

    model.add(Conv2D(64, (3,3), input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(BatchNormalization(scale=False, center=False))

    model.add(Conv2D(32, (3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2)))
    #model.add(Dropout(0.2))
    model.add(BatchNormalization(scale=False, center=False))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Dense(classes))
    model.add(Activation('softmax'))

    return model

model = build_model()

#model.compile(loss='mse', optimizer=SGD(lr=0.1), metrics=['accuracy'])	

opt = keras.optimizers.rmsprop(lr=0.0001)	#, decay=1e-6)
model.compile(loss='categorical_crossentropy',
              optimizer='Adadelta',	#opt,
              metrics=['accuracy'])

'''              
#加加加加加加加加加加加加加加加加加加加加加加加加
def generate_batch_data_random(x, y, batch_size):
    """逐步提取batch数据到显存，降低对显存的占用"""
    ylen = len(y)
    loopcount = ylen // batch_size
    while (True):
        i = np.random.randint(0,loopcount)
        yield x[i * batch_size:(i + 1) * batch_size], y[i * batch_size:(i + 1) * batch_size]
'''        

#model.fit(x_train, y_train, batch_size=600, epochs=100, validation_data=(x_vali, y_vali))
#加加加加加加加加加加加加加加加加加加加加加加加加
#history = model.fit_generator(generate_batch_data_random(x_train, y_train, 400), samples_per_epoch=400, epochs=100)
#history = model.fit_generator(generate_batch_data_random(x_train, y_train, 400), steps_per_epoch=19, epochs=100, validation_data=(x_vali, y_vali))

img_generator = ImageDataGenerator(
    rotation_range = 20,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    zoom_range = 0.2
    )
    
for i in range(100):
	print('Epoch ', i)
	print('Training...')
	progbar = Progbar(x_train.shape[0])
	batches = 0
	
	for x_batch, y_batch in img_generator.flow(x_train, y_train, batch_size=400, shuffle=True):
		# model.train_on_batch(...)应该就行了，让它等于前面的是为了把结果写在进度条上，要学会！！
		loss,train_acc = model.train_on_batch(x_batch, y_batch)	
		batches += x_batch.shape[0]
		if batches > x_train.shape[0]:
			break
		progbar.add(x_batch.shape[0], values=[('train_acc',train_acc)])
	
	#num_rand = np.random.randint(5)	
	#num_start = x_vali.shape[0] * num_rand / 5
	#num_end = num_start + x_vali.shape[0] / 5
	#test_acc = model.predict(x_vali[num_start, num_end],y_vali[num_start, num_end])
	test_acc = model.evaluate(x_vali, y_vali, batch_size=200)[1]
	print
	print('train_acc=', train_acc,', test_acc=',test_acc)
	print
	
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


#x_vali = np.array(x_all)
#y_vali = np.array(y_all)

#y_vali = hotencoder.transform(pd.DataFrame({'a':encoder.transform(y_vali)}))

y_pred = model.predict(x_vali)
y_pred[y_pred<1.0] = 0

accu = 1 - float((len(y_pred)*classes - sum((y_pred == y_vali).flatten())))/(2*len(y_pred))
print(accu)

#########

y_pred = model.predict(x_vali)
y_pred_series = pd.DataFrame(y_pred).idxmax(axis=1)
y_vali_series = pd.DataFrame(y_vali).idxmax(axis=1)

accu = float(sum(y_pred_series == y_vali_series)) / len(y_pred_series)
print accu		#与上面的model.evaluate相同！

#########

y_pred = model.predict(x_train)
y_pred_series = pd.DataFrame(y_pred).idxmax(axis=1)
y_train_series = pd.DataFrame(y_train).idxmax(axis=1)

accu = float(sum(y_pred_series == y_train_series)) / len(y_pred_series)
print accu		#与model.evaluate相同，也是0.76











