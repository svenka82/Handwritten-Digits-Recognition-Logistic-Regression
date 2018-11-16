# Author: Suhas Venkatesh Murthy
# Arizona State University

from load_dataset import read
import numpy as npy
import scipy
from matplotlib import pyplot
from scipy.sparse import csr_matrix


def softmax(z):

    e = npy.exp(z)
    denominator = npy.sum(e,axis=1).reshape((e.shape[0],1))
    return e/(1+denominator)


def sigmoid(z):

    e = npy.exp(z)
    return 1/(1+e)


def calculate_gradient(X_train,Y_train,W):

    post_prob = softmax(npy.dot(X_train,W))
    delta_weight =(npy.dot(X_train.T,(npy.subtract(Y_train,post_prob))))
    coeff = 1/float(X_train.shape[0])
    delta_weight = npy.dot(coeff,delta_weight)
    return delta_weight


def update_weights(X_train,Y_train,W):

    learning_rate = npy.exp(-4)
    x = []
    y = []

    Y_train = scipy.sparse.csr_matrix((npy.ones(Y_train.shape[0]),(Y_train,npy.array(range(Y_train.shape[0])))))
    Y_train = npy.array(Y_train.todense()).T

    for iter in [10,20,30,40,50,60,70,80,90,100]:
        for i in range(0,iter):
            delta_weight = calculate_gradient(X_train,Y_train,W)
            W = W + learning_rate * delta_weight

        x.append(iter)
        y.append(predict(W))

    j = 0
    with open('accuracies_lr.txt', 'w+') as fl:
        for i in [10,20,30,40,50,60,70,80,90,100]:
            fl.write(str(i))
            fl.write("\t")
            fl.write(str(y[j]) + "\n")
            j += 1

    pyplot.plot(x,y)
    pyplot.plot(x,y,'r+')
    pyplot.xlabel('Iterations')
    pyplot.ylabel('Accuracy')
    pyplot.savefig('accuracy_mlr.png')
    pyplot.show()


def predict(W):

    Y_test,X_test = read("testing")
    X_test = X_test.reshape((X_test.shape[0],784))
    X_test = X_test/float(255)
    Y_pred = []

    prod = npy.dot(X_test,W)
    values = softmax(prod)
    Y_pred = npy.argmax(values,axis=1)

    count = 0
    for i in range(0,X_test.shape[0]):
        if(Y_test[i] == Y_pred[i]):
            count=count+1

    return count/float(X_test.shape[0])


def driver():

    Y_train,X_train = read()
    X_train = X_train.reshape((X_train.shape[0],784))
    X_train = X_train/float(255)
    #W = npy.zeros((784,10))
    W = npy.zeros((784,10))
    update_weights(X_train,Y_train,W)


driver()