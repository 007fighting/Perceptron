# -*- encoding: utf-8 -*-

import numpy as np


class Perceptron:
    def __init__(self):
        self.W = None
        self.trainTimes = 0

    def train(self, X, Y, n, learningRate):
        '''training'''
        # random W
        row, column = X.shape
        self.W = (np.random.random(column) - 0.5) * 2

        # training for n times
        for i in range(n):
            # training
            self.trainTimes += 1
            output = np.sign(np.dot(X, self.W.T))
            gain = learningRate * ((Y - output).dot(X)) / row
            self.W += gain

            # check
            newOutput = np.sign(np.dot(X, self.W.T))
            if (newOutput == Y).all():
                break

    def getW(self):
        return self.W

    def getTrainTimes(self):
        return self.trainTimes

    def predict(self, x):
        return np.sign(np.dot(x, self.W.T))