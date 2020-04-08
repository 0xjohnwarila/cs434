import numpy as np

formatter = "{:.2f}".format
np.set_printoptions(formatter={'float_kind':formatter})

data = np.genfromtxt('housing_train.csv', delimiter=',', dtype='float')

X = np.ones(data.shape)
X[:,1:14] = data[:,:13]

Y = data[:, 13]

print(X)
print(Y)

W = np.linalg.inv(X.transpose().dot(X)).dot(X.transpose().dot(Y))
print(W)
W = np.asarray(W).reshape(-1)
Y = np.asarray(Y).reshape(-1)

wX = X

for row in wX:
    row = row * W

predicted_y = wX.sum(axis=1)

sumY = 0

for y, yH in zip(Y,predicted_y):
    sumY += (y - yH) 

print(sumY)

ase = sumY / 350

print(ase)
