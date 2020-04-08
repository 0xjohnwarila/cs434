import numpy as np

formatter = "{:.2f}".format
np.set_printoptions(formatter={'float_kind':formatter})


def train():
    data = np.genfromtxt('housing_train.csv', delimiter=',', dtype='float')

    X = np.ones(data.shape)
    X[:,1:14] = data[:,:13]

    Y = data[:, 13]


    Y = np.asarray(Y).reshape(-1)

    W = np.linalg.inv(X.transpose().dot(X)).dot(X.transpose().dot(Y))
    W = np.asarray(W).reshape(-1)

    return W, run_model(X, Y, W)

def eval(W):
    data = np.genfromtxt('housing_test.csv', delimiter=',', dtype='float')
    
    X = np.ones(data.shape)
    X[:, 1:14] = data[:,:13]

    Y = data[:, 13]

    Y = np.asarray(Y).reshape(-1)

    return run_model(X, Y, W)


def run_model(X, Y, W):
    wX = np.zeros(X.shape)

    for i in range(wX.shape[0]):
        row = np.multiply(X[i], W)
        wX[i] = row

        predicted_y = wX.sum(axis=1)

        sumY = 0

    for y, yH in zip(Y,predicted_y):
        sumY += (y - yH)**2 

    ase = sumY / Y.size
    return ase


weights, training_ase = train()
print('weights', weights)
print('training ASE:', training_ase)

testing_ase = eval(weights)
print('testing ASE:',testing_ase)
