import numpy as np	
import keras
from keras.models import Sequential 	
from keras.layers import Dense, Dropout, Activation, Flatten 	
from keras.layers import Convolution2D, MaxPooling2D 	
from keras.utils import np_utils
from matplotlib import pyplot as plt
from keras.datasets import mnist
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from keras.applications.vgg16 import VGG16
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator


import tensorflow as tf
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

sat=r'your dir'
building=r'your dir'

rootDir = sat
i=0;
dataset=list()
for dirName, subdirList, fileList in os.walk(rootDir):
    for fname in fileList:
        direc=dirName+'\%s' %fname
        image=Image.open(direc)
        image_array=np.array(image)
        dataset.append(image_array)

print('Satellite Images Loaded')

rootDir = building
i=0;
mask=list()
label=list()
for dirName, subdirList, fileList in os.walk(rootDir):
    for fname in fileList:
        direc=dirName+'\%s' %fname           
        image=Image.open(direc)
        image_array=np.array(image)
        mask_aux=image_array[:,:,2]
        mask_aux[mask_aux>=1]=1
        label.append(mask_aux)
        mask.append(image_array)
        
print('Mask Images and Labels Loaded')

X=dataset
y=list()
for j in range (0, len(label)):
    if np.mean(label[j])==0:
        y.append(0)
    else:
        y.append(1)
    
X=np.array(X)
X=X[:,:,:,:3]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255


def Vgg16():
    input_tensor = Input(shape=(224, 224, 3))
    vgg16 = VGG16(include_top=False, weights='imagenet',
                  input_tensor=input_tensor)

    top_model = Sequential()
    top_model.add(Flatten(input_shape=vgg16.output_shape[1:]))
    top_model.add(Dense(512, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(1, activation='sigmoid'))

    model = Model(input=vgg16.input, output=top_model(vgg16.output))
    for layer in model.layers[:13]:
        layer.trainable = False

    return model



def show_history(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train_accuracy', 'test_accuracy'], loc='best')
    plt.show()



if __name__ == "__main__":
    
    model = Vgg16()
    optimizer = Adam(lr=0.0001, decay=0.0)
   
    model.compile(loss='binary_crossentropy', optimizer=optimizer,
                  metrics=['accuracy'])

    earlyStopping = EarlyStopping(monitor='val_loss', patience=5, verbose=0,
                                  mode='min')
    ckpt = ModelCheckpoint('.model.hdf5', save_best_only=True,
                           monitor='val_loss', mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                       patience=3, verbose=1, epsilon=1e-4,
                                       mode='min')
    
    
    gen = ImageDataGenerator(horizontal_flip=True,
                             vertical_flip=True,
                             width_shift_range=0,
                             height_shift_range=0,
                             channel_shift_range=0,
                             zoom_range=0,
                             rotation_range=90)
    gen.fit(X_train)

    history=model.fit_generator(gen.flow(X_train, y_train, batch_size=16),
                        steps_per_epoch=len(X_train), epochs=10,
                        callbacks=[earlyStopping, ckpt, reduce_lr_loss],
                        validation_data=(X_test, y_test))

    show_history(history)
