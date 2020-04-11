import numpy as np
import argparse

formatter = "{:.2f}".format
np.set_printoptions(formatter={'float_kind':formatter})

def sigmoid(W, X):
    return 1 / (1 + np.exp(np.negative(W.transpose().dot(X))))

def train(X, Y, e, lr):
    W = np.zeros(X.shape[1])
    
    while True:
        delta = np.zeros(X.shape[1])

        for i in range(X.shape[0]):
            y_hat = sigmoid(W, X[i])
            delta = delta + (y_hat - Y[i]) * X[i]

        W = W - lr * delta

        if np.linalg.norm(delta) <= e:
            break

    return W

def get_data(filename):
    # Load data
    data = np.genfromtxt(filename, delimiter=',', dtype='float128')

    X = np.ones((data.shape[0], data.shape[1]-1))

    X[:, :256] = data[:, :256]

    Y = data[:, 256]

    Y = np.asarray(Y).reshape(-1)

    return X, Y

X, Y = get_data('usps_train.csv')

W = train(X, Y, 10, 0.0000005)
print(W)
