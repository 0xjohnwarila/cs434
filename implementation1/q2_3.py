import numpy as np
import argparse
import matplotlib.pyplot as plt

# print formatting for testing
formatter = "{:.2f}".format
np.set_printoptions(formatter={'float_kind':formatter})

# function to calculate sigmoid
def sigmoid(W, X):
    return 1 / (1 + np.exp(np.negative(W.transpose().dot(X))))

# run_model calculates the cross entropy loss
def run_model(X, Y, W):
    correct = 0

    for i in range(Y.shape[0]):
        y_hat = np.sum(X[i] * W)

        if y_hat > .5:
            y_hat = 1
        else:
            y_hat = 0

        if y_hat == Y[i]:
            correct += 1

    return correct / Y.shape[0]

# iterates batch gradient training
# returns learned weights and training data for plotting
def train(X, Y, tX, tY, e, lr, lam):
    # arrays for return
    training_loss = []
    testing_loss = []

    # Initialize weight array
    W = np.zeros(X.shape[1])
    
    # e is the number of epochs. controls how many times we iterate
    for i in range(e):
        # initialize delta array
        delta = np.zeros(X.shape[1])

        # sum all the delta values over each image in the data
        for i in range(X.shape[0]):
            y_hat = sigmoid(W, X[i])
            delta = delta + (y_hat - Y[i]) * X[i] + lam * W

        # calculate W using the learning rate and the delta value
        W = W - lr * delta

        # append learned weights to return arrays
        training_loss.append(run_model(X, Y, W))
        testing_loss.append(run_model(tX, tY, W))

    return W, training_loss, testing_loss

# loads data from csv and returns X and Y array
def get_data(filename):
    # Load data
    data = np.genfromtxt(filename, delimiter=',', dtype='float128')

    # create new array to store features
    X = np.ones((data.shape[0], data.shape[1]-1))
    X[:, :256] = data[:, :256]

    # normalization between 0 and 1
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X[i][j] = X[i][j]/255

    # copying desired outputs to Y array
    Y = data[:, 256]
    Y = np.asarray(Y).reshape(-1)

    return X, Y

# argument parsing for the command line
parser = argparse.ArgumentParser(description='Logistic Regression for Handwritten Numeric Identification')
parser.add_argument('training_data', help='csv file with training data')
parser.add_argument('testing_data', help='csv file with testing data')
args = parser.parse_args()

# reading data from arguments
X, Y = get_data(args.training_data)
tX, tY = get_data(args.testing_data)

# running above functions and printing final loss
W, rASE, tASE = train(X, Y, tX, tY, 300, 0.001, .01)
print('Training Accuracy (Cross-Entropy-Loss)', run_model(X, Y, W))
print('Testing Accuracy (Cross-Entropy-Loss)', run_model(tX, tY, W))

# plot data with matplotlib
training_plot = plt.plot(rASE, 'b', label="Training Loss")
testing_plot = plt.plot(tASE, 'g', label="Testing Loss")
plt.xlabel('Number of Iterations')
plt.ylabel('Loss Value')
plt.legend()
plt.title("Training and Testing Loss vs. Iteration Count")
plt.show()
