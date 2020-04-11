import numpy as np
import argparse
import matplotlib.pyplot as plt

formatter = "{:.2f}".format
np.set_printoptions(formatter={'float_kind':formatter})

def sigmoid(W, X):
    return 1 / (1 + np.exp(np.negative(W.transpose().dot(X))))

# run_model calculates the cross entropy loss
def run_model(X, Y, W):
    sum = 0
    for i in range(X.shape[0]):
        predictions = sigmoid(W, X[i])
        sum = sum + Y[i] * np.log(predictions) + (1 - Y[i]) * np.log(1 - predictions)
    return -sum

def train(X, Y, tX, tY, e, lr):
    training_loss = []
    testing_loss = []

    W = np.zeros(X.shape[1])
    
    for i in range(e):

        delta = np.zeros(X.shape[1])

        for i in range(X.shape[0]):
            y_hat = sigmoid(W, X[i])
            delta = delta + (y_hat - Y[i]) * X[i]

        W = W - lr * delta

        # if(i % 10 == 0):
        training_loss.append(run_model(X, Y, W))
        testing_loss.append(run_model(tX, tY, W))

    return W, training_loss, testing_loss


def get_data(filename):
    # Load data
    data = np.genfromtxt(filename, delimiter=',', dtype='float128')

    X = np.ones((data.shape[0], data.shape[1]-1))

    X[:, :256] = data[:, :256]

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X[i][j] = X[i][j]/255

    print(X)
    Y = data[:, 256]

    Y = np.asarray(Y).reshape(-1)

    return X, Y

X, Y = get_data('usps_train.csv')
tX, tY = get_data('usps_test.csv')

iterations = []

W, rASE, tASE = train(X, Y, tX, tY, 100, 0.0001)

# plot data with matplotlib

training_plot = plt.plot(rASE, 'bo--', label="Training ASE")
testing_plot = plt.plot(tASE, 'gs--', label="Testing ASE")
plt.xlabel('Number of Iterations')
plt.ylabel('ASE Value')
plt.legend()
plt.title("Training and Testing ASE vs. Iteration Count")
plt.show()
