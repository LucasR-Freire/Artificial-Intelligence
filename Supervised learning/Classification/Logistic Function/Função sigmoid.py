# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 10:22:14 2020

@author: lucas
"""

import numpy as np
import matplotlib.pyplot as plt
#%% 
def sigmoid_1(X,theta):
    X_chap = np.ones((X.shape[0],theta.shape[1])) # Contrução da matriz de ones com a quatidade de colunas igual a quantidade de colunas de theta
    XX= X_chap.T*X # Matriz X para multiplicar com a matriz coeficientes
    XX = XX.T
    #print(XX)
    X=XX
    return 1/(1 + np.exp(- np.matmul(X,theta.T)))

def sigmoid_2(X,theta):
    X_chap = np.ones((X.shape[0],theta.shape[1]))
    XX= X_chap.T*X
    XX = XX.T
    XX
    X=XX
    return np.exp(np.matmul(X,theta.T))/(1+ np.exp(np.matmul(X,theta.T)))

def test(X,theta):
    return X * theta


#%% Visualization
X_eixo = np.arange(-.5,.5,0.01)
#theta = np.random.randint(-2,11, size=(1,4))
y=sigmoid_1(X_eixo,theta)
plt.plot(X_eixo,y, label ='1/(1 +exp(-X theta)')
y=sigmoid_2(X_eixo,theta)
plt.plot(X_eixo,y, label ='exp(X theta)/(1 +exp(X theta)')
plt.legend()
plt.show()


