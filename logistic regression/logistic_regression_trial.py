import matplotlib.pyplot as plt
import numpy as np


def show_data(X , Y):
    print("Input data ")
    for i in range(X.shape[0]):
        print("Feature ",i," :",X[i]," -> ",Y[i])


def sigmoid(Z):
    return 1/(1 + np.exp(Z*-1))

def logistic_gradient(X , Y , W , B):
    m = X.shape[0] # rows
    X=np.array(X).T
    Y=np.array(Y).T
    Z = np.dot(W.T , X ) + B # linear result z
    A = sigmoid(Z)
    #backpropogation
    dZ = A-Y
    dW =np.dot(X, dZ.T)/m # its a column vector
    dB =np.sum(dZ)/m      # its sum
    return (dW,dB)

def logistic_regression(X , Y  , alpha , epochs):
    feature_length=X.shape[1]
    number_of_data=X.shape[0]
    W=np.zeros([feature_length,1]) #column vector
    B=np.zeros([1,number_of_data]) #row vector
    for i in range(epochs):
        dW,dB = logistic_gradient(X , Y , W , B)
        W-= alpha*dW
        B-= alpha*dB
    return (W , B[0][0])

def predict(xvec , w , b):
    X=np.array(xvec).T
    Z = np.dot(w.T , X ) + b # linear result z
    A = sigmoid(Z)
    return A    

X_input= np.matrix([
        [0 , 1 , 2 , 5], #feature vector 1
        [2 , 3 , 4 , 7], #feature vector 2
        [4 , 3 , 2 , 8]  #feature vector 3
        ])
Y_input= np.matrix([0,1,0]).transpose() #feature classes


show_data(X_input , Y_input)
w, b = logistic_regression(X_input , Y_input , 0.01 , 1000)
print(predict(X_input[2] , w , b))

