import numpy as np
##Sequential model type from Keras. 
##This is simply a linear stack of neural network layers, and it's perfect for the type of feed-forward CNN
from keras.models import Sequential
## importing various usefull layers
from keras.layers import Dense, Dropout , Activation,Flatten,Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
import matplotlib.pyplot as plt

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

def get_model(shape):
    ## (CONV2d,RELU) => (CONV2d ,RELU) => MaxPooling => dropout => fully connected => dropout => fullconnected_10
    input_shape=shape
    model = Sequential()
    '''keras.layers.Conv2D(filters, 
                            kernel_size, 
                            strides=(1, 1), 
                            padding='valid', 
                            data_format=None, 
                            dilation_rate=(1, 1), 
                            activation=None, 
                            use_bias=True, 
                            kernel_initializer='glorot_uniform', 
                            bias_initializer='zeros', 
                            kernel_regularizer=None, bias_regularizer=None, 
                            activity_regularizer=None, 
                            kernel_constraint=None, 
                            bias_constraint=None)
    
    '''
    model.add(Convolution2D(32 , kernel_size=(3,3), strides=(1,1) , activation='relu', input_shape=input_shape))
    model.add(Convolution2D(32 , kernel_size=(3,3), strides=(1,1) , activation='relu'))
    '''
    keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
    '''
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    '''
    compile(self, optimizer, loss, 
            metrics=None, loss_weights=None, 
            sample_weight_mode=None, weighted_metrics=None, target_tensors=None)
    '''
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

def verify_entry(X_test,model , n):
    print("Class is: ",np.argmax(model.predict(X_test[n:n+1])))
    plt.imshow(X_test[n].reshape(X_test.shape[1],X_test.shape[2]))


(X_train , Y_train),(X_test , Y_test) = load_preprocess_image_data(mnist.load_data())
model = load_model_weights("mnist_model")

'''
fit(self, 
    x=None, y=None, batch_size=None, 
    epochs=1, verbose=1, callbacks=None, 
    validation_split=0.0, validation_data=None, 
    shuffle=True, class_weight=None, sample_weight=None, 
    initial_epoch=0, steps_per_epoch=None, validation_steps=None)
'''
#model.fit(X_train, Y_train, 
#          batch_size=32, nb_epoch=10, verbose=1)

#score = model.evaluate(X_test, Y_test, verbose=1)
verify_entry(X_test , model ,100)



