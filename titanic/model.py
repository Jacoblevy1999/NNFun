import pandas as pd
import numpy as np
import random


class PerceptronNet:

    def __init__(self):
        self.eta = .3
        self.train_x = self.clean('train.csv')[0]
        self.train_y = self.clean('train.csv')[1]
        self.test_x = self.clean_test('test.csv')[0]
        self.passenger_ids = self.clean_test('test.csv')[1]
        features = len(self.train_x[0])
        w = []
        for i in range(features):
            w.append(0)
        self.wts = w

    def clean(self, data):
        t = pd.read_csv(data)
        t.dropna(inplace=True)
        y = t['Survived']
        to_drop = ['Name', 'Survived',
                   'Ticket', 'Cabin', 'PassengerId']
        t.drop(to_drop, inplace=True, axis=1)
        t.replace({'Embarked': 'S'}, 1, inplace=True)
        t.replace({'Embarked': 'C'}, 2, inplace=True)
        t.replace({'Embarked': 'Q'}, 3, inplace=True)
        t.replace({'Sex': 'male'}, 1, inplace=True)
        t.replace({'Sex': 'female'}, 0, inplace=True)
        x = t.values.tolist()
        new = []
        for i in x:
            new.append([1]+i)
        return (new, y.values.tolist())

    def clean_test(self, data):
        t = pd.read_csv(data)
        t.fillna(method='pad', inplace=True)
        p_ids = t['PassengerId']
        to_drop = ['Name',
                   'Ticket', 'Cabin', 'PassengerId']
        t.drop(to_drop, inplace=True, axis=1)
        t.replace({'Embarked': 'S'}, 1, inplace=True)
        t.replace({'Embarked': 'C'}, 2, inplace=True)
        t.replace({'Embarked': 'Q'}, 3, inplace=True)
        t.replace({'Sex': 'male'}, 1, inplace=True)
        t.replace({'Sex': 'female'}, 0, inplace=True)
        x = t.values.tolist()
        new = []
        for i in x:
            new.append([1]+i)
        return (new, p_ids.values.tolist())

    def feedforward(self, example, result, wts):
        x = np.dot(example, wts)
        if x > 20:
            return result-1
        elif x < -20:
            return result-0
        sgm = 1/(1 + np.exp(-x))
        return result-sgm

    def train(self, times, eta=0):
        if eta == 0:
            eta = self.eta
        train = self.train_x
        val = self.train_y
        for p in range(times):
            for i in range(len(train)):
                wts = self.wts
                ex = train[i]
                diff = self.feedforward(ex, val[i], wts)
                new = []
                for j in range(len(train[i])):
                    w_new = wts[j]+(diff*eta*ex[j])
                    new.append(w_new)
                self.wts = new
        return self.wts

    def pct_correct(self, times, eta=0):
        if eta == 0:
            eta = self.eta
        wts = self.train(times, eta)
        test = self.train_x
        pred = []
        for i in test:
            v = np.dot(i, wts)
            if v >= 0:
                pred.append(1)
            else:
                pred.append(0)
        total = len(pred)
        correct = 0
        for i in range(len(pred)):
            if pred[i] == self.train_y[i]:
                correct += 1
        return correct/total

    def refresh(self):
        w = []
        for i in range(len(self.train_x[0])):
            w.append(0)
        self.wts = w

    def predictions(self, times, eta=0):
        if eta == 0:
            eta = self.eta
        wts = self.train(25, eta)
        test = self.test_x
        pred = []
        for i in test:
            v = np.dot(i, wts)
            if v >= 0:
                pred.append(1)
            else:
                pred.append(0)
        p_ids = self.passenger_ids
        df = pd.DataFrame(list(zip(p_ids, pred)),
                          columns=['PassengerId', 'Survived'])
        df.to_csv(
            r'/Users/jacoblevy/Desktop/titanic/soln.csv', index=False)


# class DoubleLayer(PerceptronNet):
#     """fill in later"""
