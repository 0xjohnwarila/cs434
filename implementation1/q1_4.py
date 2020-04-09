import numpy as np
import argparse

def run(training_data, testing_data):
    rX, rY = get_data(training_data)
    tX, tY = get_data(testing_data)
    rX = add_features(rX)
    # print(rY, tY)

# take array and return array with two extra columns of
# random sampled values
def add_features(X):
    # create empty array 2 wider than provided array
    new_array = np.ones((X.shape[0], X.shape[1]+2))

    # copy provided array over to new array
    new_array[:, :X.shape[1]] = X



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

parser = argparse.ArgumentParser(description='ASE testing with added random features')
parser.add_argument('training_data', help='csv file with training data')
parser.add_argument('testing_data', help='csv file with testing data')
args = parser.parse_args()

run(args.training_data, args.testing_data)
