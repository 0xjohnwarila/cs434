import numpy as np
import argparse
import matplotlib.pyplot as plt

# Formating for printing
formatter = "{:.2f}".format
np.set_printoptions(formatter={'float_kind':formatter})

def run(training_data, testing_data):
    # put csv info into arrays
    rX, rY = get_data(training_data)
    tX, tY = get_data(testing_data)

    # arrays to stores plot data
    x_axis = []
    rASE = []
    tASE = []

    for i in range(10):
        # create array to track new features
        x_axis.append((i+1)*2)

        # add random sampled data into two new columns on training X matrix
        # and testing X matrix
        rX = add_features(rX)
        tX = add_features(tX)

        # calculate the weights for each the training data
        W = calculate_weights(rX, rY)

        # add results to array of all ASE results
        rASE.append(run_model(rX, rY, W))
        tASE.append(run_model(tX, tY, W))

    # plot data with matplotlib
    training_plot = plt.plot(x_axis, rASE, 'bo--', label="Testing ASE")
    testing_plot = plt.plot(x_axis, tASE, 'gs--', label="Training ASE")
    plt.xlabel('Added Features')
    plt.ylabel('ASE Value')
    plt.legend()
    plt.title("Training and Testing ASE vs. Features Added")
    plt.show()

def calculate_weights(X, Y):
    # Calculate the weights. (X^T X)^-1 X^T Y
    W = np.linalg.inv(X.transpose().dot(X)).dot(X.transpose().dot(Y))
    # Reshape for ease of use
    W = np.asarray(W).reshape(-1)

    return W

# eval gets the ASE for the testing data
def eval(W, testing_data):
    # Get the data from the testing file
    X, Y = get_data(testing_data)

    # Run the model and return ASE
    return run_model(X, Y, W)

# take array and return array with two extra columns of
# random sampled values
def add_features(X):
    # create empty array 2 wider than provided array
    new_array = np.ones((X.shape[0], X.shape[1]+2))

    # copy provided array over to new array
    new_array[:, :X.shape[1]] = X

    for i in range(new_array.shape[0]):
        # for the last two columns, sample from a normal dist and add to array
        new_array[i][new_array.shape[1]-1] = np.random.normal(10,5,1)
        new_array[i][new_array.shape[1]-2] = np.random.normal(10,5,1)

    return new_array

def get_data(filename):
    # Load data from csv file
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

parser = argparse.ArgumentParser(description='ASE testing with added random features')
parser.add_argument('training_data', help='csv file with training data')
parser.add_argument('testing_data', help='csv file with testing data')
args = parser.parse_args()

# run above code
run(args.training_data, args.testing_data)
