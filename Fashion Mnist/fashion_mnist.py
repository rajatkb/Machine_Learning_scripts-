# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 06:43:45 2018

@author: Rajat
"""

import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist as dataset
from keras.models import Sequential
from keras.layers import Dense, Dropout , Activation,Flatten,Convolution2D, MaxPooling2D ,AveragePooling2D
from keras.utils import np_utils



def load_preprocess_image_data(data):
    (X_train , Y_train),(X_test , Y_test) = data
    # making data of type (m , h, w , c)
    X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1) 
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2],1) 
    # making data normalize
    X_norm_train = X_train.astype('float32')/255
    X_norm_test = X_test.astype('float32')/255
    # making classes having one hot encoding
    n_class = np.unique(Y_train).shape[0]
    Y_encoded_train = np_utils.to_categorical(Y_train,n_class)
    Y_encoded_test = np_utils.to_categorical(Y_test,n_class)
    return ((X_norm_train , Y_encoded_train),(X_norm_test,Y_encoded_test))

def detection_model(shape):
    # Conv2d(3,3,c = 64) => Conv2d(4,4,c=128) => Conv2d(2,2,256) => AveragePool(2,2) =>dropoout=> faltten=>Dense=>dropout
    # => dense => dense =>output
    model = Sequential()
    model.add(Convolution2D(32 , kernel_size=(3,3), strides=(1,1) , padding='same' , activation='relu', input_shape=shape))
    model.add(Convolution2D(64 , kernel_size=(4,4), strides=(1,1) , padding='same' , activation='relu', input_shape=shape))
    model.add(Convolution2D(128 , kernel_size=(4,4), strides=(1,1) , padding='valid' , activation='relu', input_shape=shape))
    model.add(AveragePooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

def save_model(model,name):
     model_json = model.to_json()
     with open(name+".json", "w") as json_file:
         json_file.write(model_json)
     model.save_weights(name+".h5")
    
def load_model_weights(name):
    from keras.models import model_from_json
    json_file = open(name+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(name+".h5")
    loaded_model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return loaded_model
    
def verify_entry(X_test,Y_test,model , n):
    print("Actual Class is:",np.argmax(Y_test[n:n+1]))
    print("Predicted Class is: ",np.argmax(model.predict(X_test[n:n+1])))
    plt.imshow(X_test[n].reshape(X_test.shape[1],X_test.shape[2]))

    
(X_train , Y_train),(X_test , Y_test) = load_preprocess_image_data(dataset.load_data())
model = detection_model(X_train[0].shape)
model.summary()
model.fit(X_train, Y_train, 
          batch_size=32, nb_epoch=10, verbose=1)
score = model.evaluate(X_test, Y_test, verbose=1)
#save_model(model,"fashion_mnist")

