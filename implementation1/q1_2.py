import numpy as np
import argparse

# Formating for printing
formatter = "{:.2f}".format
np.set_printoptions(formatter={'float_kind':formatter})

# train generates a set of weights from the training data
def train(training_data):
    # Get the data from the training file. X is params, Y is goal
    X, Y = get_data(training_data)

    # Calculate the weights. (X^T X)^-1 X^T Y
    W = np.linalg.inv(X.transpose().dot(X)).dot(X.transpose().dot(Y))
    # Reshape for ease of use
    W = np.asarray(W).reshape(-1)

    # Return the weights, and the Average Squared Error (ASE)
    return W, run_model(X, Y, W)

# eval gets the ASE for the testing data
def eval(W, testing_data):
    # Get the data from the testing file
    X, Y = get_data(testing_data)

    # Run the model and return ASE
    return run_model(X, Y, W)

# get_data is a helper to build X and Y from a file
def get_data(filename):
    # Load from csv file
    data = np.genfromtxt(filename, delimiter=',', dtype='float')

    # Build an array of ones
    X = np.ones(data.shape)
    # Set all but the first column to be equal to the first 13 columns of the
    # data
    X[:, 1:14] = data[:,:13]

    # Set Y to be the 14th column of data
    Y = data[:, 13]

    # Reshaping for ease of use
    Y = np.asarray(Y).reshape(-1)
    
    return X, Y

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

parser = argparse.ArgumentParser(description='Linear regression for housing price')
parser.add_argument('training_data', help='csv file with training data')
parser.add_argument('testing_data', help='csv file with testing data')
args = parser.parse_args()
# Running the above methods
weights, training_ase = train(args.training_data)
print('weights', weights)
print('training ASE:', training_ase)

testing_ase = eval(weights, args.testing_data)
print('testing ASE:',testing_ase)
