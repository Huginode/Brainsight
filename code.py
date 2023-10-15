import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# created dataset
X, y =

# ofc reshape y
y = y.reshape((y.shape[0], 1))

#graph plotting

# def init function
def init(X):
    W = np.random.randn(X.shape[1], 1)
    b = np.random.randn(1)
    return (W, b)

# def the model
def model(X, W, b):
    Z = X.dot(W) + b
    A = 1/(1 + np.exp(-Z))
    return A

# def cost function
def logLoss(A, y):
    return 1 / len(y) * np.sum(-y * np.log(A) - (1 - y) * np.log(1 - A) )


# def gradient function
def gradients (A, x, y):
    dW = 1 / len(y) * np.dot(X.T, A - y)
    db = 1 / len(y) * np.sum(A - y)
    return (dW, db)


# update function
def update(dW, db, W, b, learningRate):
    W = W - learningRate * dW
    b = b - learningRate * db
    return (W, b)

# True algorithm
def artificialNeuron(X, y, learningRate=0.1, nIter=100):
    # init W and b parameters
    W, b = init(X)
    loss = []

    for i in range(nIter):
        A = model(X, W, b)
        loss.append(logLoss(A, y))
        dW, db = gradients(A, X, y)
        W, b = update(dW, db, W, b, learningRate)

    yPred = predict(X, W, b)
    print(accuracy_score(y, yPred))

    plt.plot(loss)
    plt.show

    return (W, b)


# Predicting function
def predict(X, W, b):
    A = model(X, W, b)
    return A >= 0.5


W, b = artificialNeuron(X, y)
plt.show()

# args to draw the descision line
x0 = np.linspace(0, 0, 100)
x1 = (-W[0] * x0 -b) / W[1]

plt.scatter(X[:, 0], X[:, 1], c=y, cmap='summer')
plt.plot(x0, x1, c='orange', lw=3)
plt.show()
predict(X, W, b)
