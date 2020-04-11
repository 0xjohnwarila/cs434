import numpy as np
import argparse
import matplotlib.pyplot as plt

formatter = "{:.2f}".format
np.set_printoptions(formatter={'float_kind':formatter})

def sigmoid(W, X):
    return 1 / (1 + np.exp(np.negative(W.transpose().dot(X))))

# run_model calculates the ASE
def run_model(X, Y, W):
    # Place holder matrix that will hold the X rows times the weights
    wX = np.zeros(X.shape)

    # For every row in X, multiply elementwise by W
    for i in range(wX.shape[0]):
        row = np.multiply(X[i], W)
        wX[i] = row

        # Get the predicted Y by summing the row
        predicted_y = wX.sum(axis=1)

    sumY = 0
    # Sum the squared error between Y and predicted Y
    for y, yH in zip(Y,predicted_y):
        sumY += (y - yH)**2 

    # Average for the average squared error
    ase = sumY / Y.size
    return ase

def train(X, Y, tX, tY, e, lr):
    W = np.zeros(X.shape[1])
    
    i = 0
    while True:
        # track number of iterations
        i += 1
        iterations.append(i)

        delta = np.zeros(X.shape[1])

        for i in range(X.shape[0]):
            y_hat = sigmoid(W, X[i])
            delta = delta + (y_hat - Y[i]) * X[i]

        W = W - lr * delta

        rASE.append(run_model(X, Y, W))
        tASE.append(run_model(tX, tY, W))

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
tX, tY = get_data('usps_test.csv')

iterations = []
rASE = []
tASE = []

W = train(X, Y, tX, tY, 10, 0.0000005)

# plot data with matplotlib
training_plot = plt.plot(iterations, rASE, 'bo--', label="Training ASE")
testing_plot = plt.plot(iterations, tASE, 'gs--', label="Testing ASE")
plt.xlabel('Number of Iterations')
plt.ylabel('ASE Value')
plt.legend()
plt.title("Training and Testing ASE vs. Iteration Count")
plt.show()
