import pandas as pd
import numpy as np
import random


class PerceptronNet:

    def __init__(self):
        self.eta = .1
        self.train_x = self.clean('train.csv')[0]
        self.train_y = self.clean('train.csv')[1]
        self.test_x = self.clean_test('test.csv')
        features = len(self.train_x[0])
        w = []
        for i in range(features):
            w.append(0)
        self.wts = w

    def clean(self, data):
        t = pd.read_csv(data)
        t.dropna(inplace=True)
        y = t['Survived']
        to_drop = ['SibSp', 'Embarked', 'Name',
                   'Ticket', 'Cabin', 'PassengerId', 'Survived']
        t.drop(to_drop, inplace=True, axis=1)
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
        to_drop = ['SibSp', 'Embarked', 'Name',
                   'Ticket', 'Cabin', 'PassengerId']
        t.drop(to_drop, inplace=True, axis=1)
        t.replace({'Sex': 'male'}, 1, inplace=True)
        t.replace({'Sex': 'female'}, 0, inplace=True)
        x = t.values.tolist()
        new = []
        for i in x:
            new.append([1]+i)
        return new

    def feedforward(self, example, result, wts):
        dot = np.dot(example, wts)
        if dot >= 0:
            h = 1
        else:
            h = 0
        return result-h

    def train(self, times):
        train = self.train_x
        val = self.train_y
        for p in range(times):
            for i in range(len(train)):
                wts = self.wts
                ex = train[i]
                diff = self.feedforward(ex, val[i], wts)
                new = []
                for j in range(len(train[i])):
                    w_new = wts[j]+(diff*self.eta*ex[j])
                    new.append(w_new)
                self.wts = new
        return self.wts

    def predictions(self):
        wts = self.train(100)
        test = self.test_x
        pred = []
        for i in test:
            v = np.dot(i, wts)
            if v >= 0:
                pred.append(1)
            else:
                pred.append(0)
        p_ids = []
        for i in range(1, len(pred)+1):
            p_ids.append(i)
        df = pd.DataFrame(list(zip(p_ids, pred)),
                          columns=['PassengerId', 'Survived'])
        df.to_csv(
            r'/Users/jacoblevy/Desktop/titanic/soln.csv', index=False)
