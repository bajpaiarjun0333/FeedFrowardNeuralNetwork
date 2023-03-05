import numpy as np 
import pandas as pd
import math

#dataset loading 
from keras.datasets import fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train=x_train.reshape(x_train.shape[0],(x_train.shape[1]*x_train.shape[2]))
x_test=x_test.reshape(x_test.shape[0],(x_test.shape[1]*x_test.shape[2]))
x_train=x_train/255
x_test=x_test/255



class NeuralNetwork:
    def __init__(self):
        self.w=[]
        self.b=[]
        self.a=[]
        self.h=[]
        self.wd=[]
        self.ad=[]
    
    def activations(activation,z):
        if activation=='sigmoid':
            return 1/(1+np.exp(-z))
        elif activation=='relu':
            return z*(z>0)
        elif activation=='tanh':
            return (np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))
        elif activation=='softmax':
            for i in range(z.shape[0]):
                sum=0
                for j in range(z.shape[1]):
                    sum=sum+np.exp(z[i][j])
            z[i]=np.exp(z[i])/sum
        return z
    
    def activations_derivative(activation,z):
        if activation=='sigmoid':
            return np.multiply((1/(1+np.exp(-z))),(1-(1/(1+np.exp(-z)))))
        elif activation=='relu':
            return z*(z>0)
        elif activation=='tanh':
            y=(np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))
            return 1-np.square(y)
        
    def loss_function(loss_fn,yhat,y_train):
        if loss_fn=='cross_entropy':
            sum=0
            for i in range(y_train.shape[0]):
                sum+=-((np.log2(yhat[i][y_train[i]])))
            return sum
        if loss_fn=='mean_square':
            return (yhat-y_train)**2


