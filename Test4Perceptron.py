# -*- encoding: utf-8 -*-

from Perceptron import *
import numpy as np
import matplotlib.pyplot as plt


def test():
    # training data
    X = np.array([[1, 3, 3], [1, 4, 3], [1, 1, 1]])
    Y = np.array([1, 1, -1])
    learningRate = 0.1
    learningTimes = 100

    # training
    perceptron = Perceptron()
    perceptron.train(X, Y, learningTimes, learningRate)
    W = perceptron.getW()
    trainTimes = perceptron.getTrainTimes()
    print W
    print trainTimes

    # plot the training data
    X1 = [3, 4]
    Y1 = [3, 3]
    X2 = [1]
    Y2 = [1]
    k = -W[1] / W[2]
    d = -W[0] / W[2]
    x = np.linspace(0, 5)   # generate arithmetic sequence

    # plot
    plt.figure()
    plt.plot(x, x*k+d, 'r')
    plt.plot(X1, Y1, 'bo')
    plt.plot(X2, Y2, 'yo')
    plt.show()

    # predict
    test = np.array([[1, 2, 3], [1, 6, 2], [1, 3, 3], [1, 7, 5], [1, 5, 7], [1, 9, 2]])
    testResult = perceptron.predict(test)
    print testResult
    testX1 = []
    testY1 = []
    testX2 = []
    testY2 = []
    for i in range(len(testResult)):
        if testResult[i] >= 0:
            testX1.append(test[i][1])
            testY1.append(test[i][2])
        else:
            testX2.append(test[i][1])
            testY2.append(test[i][2])
    plt.figure()
    plt.plot(x, x*k+d, 'r')
    plt.plot(testX1, testY1, 'bo')
    plt.plot(testX2, testY2, 'yo')
    plt.show()

if __name__ == '__main__':
    test()