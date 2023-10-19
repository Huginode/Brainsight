import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# setting the path to the directory containing the pics
pathN = 'dataset/no'
pathY = 'dataset/yes'

# appending the the datasets lists
trainingDataN = []
trainingDataY = []
for img in os.listdir(pathN):
    pic = cv2.imread(os.path.join(pathN, img))
    pic = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
    pic = cv2.resize(pic, (80, 80))
    trainingDataN.append([pic])

for img in os.listdir(pathY):
    pic = cv2.imread(os.path.join(pathY, img))
    pic = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
    pic = cv2.resize(pic, (80, 80))
    trainingDataY.append([pic])

# concatenate the two lists for the dataset
trainingData = trainingDataN + trainingDataY

# converting the list to numpy array
X_train = np.array(trainingData)
X_train = np.squeeze(X_train)

# creating y axis using the trainingDataN and Y for labeling
y_train1 = np.zeros(len(trainingDataN))
y_train = np.ones(len(trainingDataY))
y_train = np.concatenate((y_train1, y_train), axis=0)
y = y_train.reshape((y_train.shape[0], 1))

# flatten the pixels to transform the X data in 2d. Otherwise it doesn't work
X = X_train.reshape(X_train.shape[0], -1)

# normalisation des donnes les photos sont en 8 bits donc le min est 0 donc la fonction se simplifie
X = X_train.reshape(X_train.shape[0], -1) / X_train.max()

# def init functiond
def init(X):
    W = np.random.randn(X.shape[1], 1)
    b = np.random.randn(1)
    return (W, b)


# def the model with sigmoid function
def model(X, W, b):
    Z = X.dot(W) + b
    A = 1/(1 + np.exp(-Z))
    return A


# def cost function
def logLoss(A, y):
    epsilon = 1e-15
    return 1 / len(y) * np.sum(-y * np.log(A + epsilon) - (1 - y) * np.log(1 - A + epsilon))


# def gradient function
def gradients(A, X, y):
    dW = 1 / len(y) * np.dot(X.T, A - y.T)
    db = 1 / len(y) * np.sum(A - y)
    return (dW, db)


# update function
def update(dW, db, W, b, learningRate):
    W = W - learningRate * dW
    b = b - learningRate * db
    return (W, b)


# True algorithm
def artificialNeuron(X, y, learningRate, nIter):
    # init W and b parameters
    W, b = init(X)
    loss = []
   #acc = []

    for i in tqdm(range(nIter)):
        # loading the model
        A = model(X, W, b)

        # make code not so slow
        if i %25 == 0:
            # cost calculation
            loss.append(logLoss(A, y))
            # accuracy calculation
            yPred = predict(X, W, b)
            #acc.append(accuracy_score(y, yPred))

        # updating the parameters
        dW, db = gradients(A, X, y)
        W, b = update(dW, db, W, b, learningRate)

    # plotting the graphs
   # plt.figure(figsize=(12, 4))
   # plt.subplot(1, 2, 1)
    plt.plot(loss)
   # plt.subplot(1, 2, 1)
   # plt.plot(acc)
    plt.show()

    return (W, b)


# Predicting function
def predict(X, W, b):
    A = model(X, W, b)
    return A >= 0.5


W, b = artificialNeuron(X, y, learningRate=0.01, nIter=1000)

# args to draw the descision line
#x0 = np.linspace(0, 0, 100)
#x1 = (-W[0] * x0 - b) / W[1]

#plt.scatter(X[:, 0], X[:, 1], c=y_train, cmap='summer')
#plt.plot(x0, x1, c='orange', lw=3)
#predict(X, W, b)
#plt.show()
