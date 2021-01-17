from tensorflow import keras
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import glob 
import numpy as np
from os import listdir,makedirs
from os.path import isfile,join
from numpy import *
from PIL import Image
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import random
import cv2
import os 

path_2="/Users/UDAY KUMAR/Desktop/major_project/CataractDetection-master/dataset"

local_path=path_2+"/1_normal"
files = [f for f in listdir(local_path) if isfile(join(local_path,f))]

random.shuffle(files) 
files=files[0:100]
for i in files:
	originalImage = cv2.imread(local_path+"/"+i)
	dstPath = join(path_2+"/training_data",i)
	cv2.imwrite(dstPath,originalImage)
# print(len(files))
local_path=path_2+"/2_cataract"
files_1 = [f for f in listdir(local_path) if isfile(join(local_path,f))]
for i in files_1:
	files.append(i)
	originalImage = cv2.imread(local_path+"/"+i)
	dstPath = join(path_2+"/training_data",i)
	cv2.imwrite(dstPath,originalImage)

local_path=path_2+"/2_glaucoma"
files_2 = [f for f in listdir(local_path) if isfile(join(local_path,f))]
files_2=files_2[0:100]
for i in files_2:
	files.append(i)
	originalImage = cv2.imread(local_path+"/"+i)
	dstPath = join(path_2+"/training_data",i)
	cv2.imwrite(dstPath,originalImage)

local_path=path_2+"/3_retina_disease"
files_3 = [f for f in listdir(local_path) if isfile(join(local_path,f))]
for i in files_3:
	files.append(i)
	originalImage = cv2.imread(local_path+"/"+i)
	dstPath = join(path_2+"/training_data",i)
	cv2.imwrite(dstPath,originalImage)

listing=os.listdir(path_2)
# for image in files:
# 	img = cv2.imread(os.path.join(path_2,image))
# 	imlist.append(img.flatten())
immatrix=np.array([np.array(cv2.imread(os.path.join(path_2,image))).flatten()
	for image in files])
num_samples=400


label=np.ones((num_samples,),dtype=int)
label[0:100]=0
label[100:200]=1
label[200:301]=2
label[300:]=3
data,label=shuffle(immatrix,label,random_state=2)
channels = 3

dataset = np.ndarray(shape=(len(files), channels,200,200),
                     dtype=np.float32)
i=0
for _file in files:
	if i<100:
		img = load_img(path_2+"/1_normal"+ "/" + _file)
		# print(img)
		x = img_to_array(img)  
		#print(x)
		x = x.reshape((3,2000,2000))
		dataset[i]=x
	elif i>=100 and i<200:
		img = load_img(path_2+"/2_cataract"+ "/" + _file)
		# print(img)
		x = img_to_array(img)  
		# print(x)
		x = x.reshape((3,200,200))
		dataset[i]=x
	elif i>=200 and i<300:
		img = load_img(path_2+"/2_glaucoma"+ "/" + _file)
		# print(img)
		x = img_to_array(img)  
		# print(x)
		x = x.reshape((3,200,200))
		dataset[i]=x
	else:
		img = load_img(path_2+"/3_retina_disease"+ "/" + _file)
		# print(img)
		x = img_to_array(img)  
		# print(x)
		x = x.reshape((3,200,200))
		dataset[i]=x
	i+=1
from sklearn.model_selection import train_test_split
# print(dataset)
#Splitting 
X_train, X_test, y_train, y_test = train_test_split(dataset,label, test_size=0.2, random_state=33)


# print(len(X_train))
# print(len(X_test))
batch_size=32

nb_classess=4

nb_epoch=20
img_rows,img_cols=10,32

img_channels=1

nb_filters=32

nb_pool=2

nb_conv=3


# X,y=(training_data[0],training_data[1])

# print(len(training_data[0]))
# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=4)
# print(X_train.shape)
# # plt.imshow(X_train[0].reshape(300,400))
# # print(len(X_train))
# X_train=np.array([i for i in X_train]).reshape(-1,32,10, 3) 
# y_train = [i for i in y_train] 
# X_test = np.array([i for i in X_test]).reshape(-1,32,10, 3) 
# y_test = [i for i in y_test] 
# # image = array(img).reshape(1, 64,64,3)




from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
from keras.layers import Activation, Dropout, Flatten, Dense, GlobalMaxPooling2D, Conv2D, MaxPooling2D
from keras.callbacks import CSVLogger
# from livelossplot.keras import PlotLossesCallback
# import efficientnet.keras as efn
model = Sequential()

input_shape = (200,200, 3)
model.add(Conv2D(32, 3, 3, border_mode='same', input_shape=input_shape, activation='relu'))
model.add(Conv2D(32, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, 3, 3, border_mode='same', activation='relu'))
model.add(Conv2D(64, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, 3, 3, border_mode='same', activation='relu'))
model.add(Conv2D(128, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, 3, 3, border_mode='same', activation='relu'))
model.add(Conv2D(256, 3, 3, border_mode='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
            optimizer=RMSprop(lr=0.0001),
            metrics=['accuracy'])



training_data_generator = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.2)
validation_data_generator = ImageDataGenerator(rescale=1./255)
test_data_generator = ImageDataGenerator(rescale=1./255)

# IMAGE_WIDTH=200
# IMAGE_HEIGHT=200
IMAGE_SIZE = 200
IMAGE_WIDTH, IMAGE_HEIGHT = IMAGE_SIZE, IMAGE_SIZE
EPOCHS = 20
BATCH_SIZE = 32
TEST_SIZE = 30
training_data_dir=path_2+"/training_data"

train_datagen = ImageDataGenerator(rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2) # set validation split

train_generator = train_datagen.flow_from_directory(
    training_data_dir,
    target_size=(img_rows, img_cols),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training') # set as training data

validation_generator = train_datagen.flow_from_directory(
    training_data_dir , # same directory as training data
    target_size=(img_rows, img_cols),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation') # set as validation data



model.fit_generator(
    train_generator,
    steps_per_epoch = train_generator.samples // BATCH_SIZE,
    validation_data = validation_generator, 
    validation_steps = validation_generator.samples // BATCH_SIZE,
    epochs = EPOCHS)