import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split

from tools.arff import Arff
import matplotlib.pyplot as plt


class PerceptronClassifier(BaseEstimator,ClassifierMixin):

    def __init__(self, lr=.1, shuffle=False, epochs=0):
        """ Initialize class with chosen hyperparameters.
        Args:
            lr (float): A learning rate / step size.
            shuffle: Whether to shuffle the training data each epoch. DO NOT SHUFFLE for evaluation / debug datasets.
        """
        self.lr = lr
        self.shuffle = shuffle
        self.epochs = epochs
        self.accuracy_test = 0
        self.epochs_run = 0

    def fit(self, patterns, targets, initial_weights = []):
        """ Fit the data; run the algorithm and adjust the weights to find a good solution

        Args:
            patterns (array-like): A 2D numpy array with the training data, excluding targets
            targets (array-like): A 2D numpy array with the training targets
            initial_weights (array-like): allows the user to provide initial weights

        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)

        """

        # Split training and testing set
        self.patterns_train, self.patterns_test, self.targets_train, self.targets_test = \
            train_test_split(patterns, targets, test_size=0.33, random_state=42)
        # self.patterns_train = patterns
        # self.patterns_test = patterns
        # self.targets_train = targets
        # self.targets_test = targets

        # Initializations
        self.num_train_inputs = self.patterns_train.shape[1]
        if not initial_weights:
            self.initialize_weights()

        # Add bias
        self.patterns_train = np.append(self.patterns_train, np.ones(self.patterns_train.shape[0]).reshape(-1,1), axis=1)
        self.patterns_test = np.append(self.patterns_test, np.ones(self.patterns_test.shape[0]).reshape(-1,1), axis=1)

        # Calculate weights for each pattern
        if self.epochs == 0:
            max_epochs_small_accuracy_change = 5
            num_epoch_small_accuracy_change = 0
            small_accuracy_change = 0.01
            max_accuracy = 0
            while num_epoch_small_accuracy_change < max_epochs_small_accuracy_change:
                if self.shuffle:
                    self.shuffle_data()

                # Training
                for i, pattern in enumerate(self.patterns_train):
                    net = np.dot(pattern, self.weights)
                    output = np.where(net > 0, 1, 0)
                    self.weights += (self.targets_train[i] - output) * pattern.reshape(-1, 1) * self.lr

                # Training Accuracy
                output_train = self.predict(self.patterns_train)
                self.accuracy_train = self.score(self.patterns_train, self.targets_train, output_train)

                # Test Accuracy
                if self.accuracy_test == 0:
                    output_test = self.predict(self.patterns_test)
                    self.accuracy_test = self.score(self.patterns_test, self.targets_test, output_test)
                else:
                    old_accuracy = self.accuracy_test
                    output_test = self.predict(self.patterns_test)
                    self.accuracy_test = self.score(self.patterns_test, self.targets_test, output_test)
                    if self.accuracy_test >= old_accuracy:
                        max_accuracy = self.accuracy_test

                    if (self.accuracy_test - max_accuracy) < small_accuracy_change:
                        num_epoch_small_accuracy_change += 1
                    else:
                        num_epoch_small_accuracy_change = 0

                self.epochs_run += 1

        else:
            for _ in range(self.epochs):
                if self.shuffle:
                    self.shuffle_data()
                for i, pattern in enumerate(self.patterns_train):
                    net = np.dot(pattern, self.weights)
                    output = np.where(net > 0, 1, 0)
                    self.weights += (self.targets_train[i] - output) * pattern.reshape(-1, 1) * self.lr
            self.epochs_run = self.epochs

            # Training Accuracy
            output_train = self.predict(self.patterns_train)
            self.accuracy_train = self.score(self.patterns_train, self.targets_train, output_train)

            # Test Accuracy
            output_test = self.predict(self.patterns_test)
            self.accuracy_test = self.score(self.patterns_test, self.targets_test, output_test)



        return self

    def predict(self, X):
        """ Predict all classes for a dataset X

        Args:
            X (array-like): A 2D numpy array with the test data, excluding targets

        Returns:
            array, shape (n_samples,)
                Predicted target values per element in X.
        """
        o = np.where(np.dot(X, self.weights) > 0, 1, 0)

        return o


    def initialize_weights(self):
        """ Initialize weights for perceptron. Don't forget the bias!

        Returns:

        """
        self.weights = np.random.random(self.num_train_inputs + 1).reshape(-1, 1)
        # self.weights = np.zeros(self.num_train_inputs + 1).reshape(-1, 1)

        return self.weights

    def score(self, X, y, o):
        """ Return accuracy of model on a given dataset. Must implement own score function.

        Args:
            X (array-like): A 2D numpy array with data, excluding targets
            y (array-like): A 2D numpy array with targets

        Returns:
            score : float
                Mean accuracy of self.predict(X) wrt. y.
        """
        score = 0
        for idx, target in enumerate(y):
            if target == o[idx]:
                score += 1
        score = score / y.size

        return score

    def shuffle_data(self):
        """ Shuffle the data! This _ prefix suggests that this method should only be called internally.
            It might be easier to concatenate X & y and shuffle a single 2D array, rather than
             shuffling X and y exactly the same way, independently.
        """
        data = np.column_stack((self.patterns_train, self.targets_train))
        np.random.shuffle(data)
        self.patterns_train = data[:, 0:-1]
        self.targets_train = data[:, -1]

        pass

    def get_weights(self):
        return self.weights.reshape(1,-1)

# Data files
data = Arff("../data/perceptron/new_vote.arff", label_count=1)
# data = Arff("../data/perceptron/evaluation/data_banknote_authentication.arff", label_count=1)
# data = Arff("../data/perceptron/debug/linsep2nonorigin.arff", label_count=1)
# data = Arff("../data/perceptron/debug/linearlysep.arff", label_count=1)
# data = Arff("../data/lin_sep.arff", label_count=1)
# data = Arff("../data/not_lin_sep.arff", label_count=1)

patterns = data[:, 0:-1]
targets = data[:, -1]

pcn = PerceptronClassifier(lr=.1,shuffle=False)
pcn.fit(patterns,targets)
print("Test Accuracy = [{:.2f}]".format(pcn.accuracy_test))
print("Training Accuracy = [{:.2f}]".format(pcn.accuracy_train))
print("Final Weights =", pcn.get_weights())
print("Epochs Run =", pcn.epochs_run)

## Graph a function using matplotlib
fig, ax = plt.subplots()
y_func = lambda x: (-pcn.weights[0] * x + pcn.weights[2]) * (1/pcn.weights[1])
x_s = data[:,0]
y_s = data[:,1]
labels = data[:, -1]
red = np.array([.8, .1, 0])
blue = np.array([0, 0.1, 0.8])
colors = []
x_s_red = []
y_s_red = []
x_s_blue = []
y_s_blue = []
for label in labels:
    if label == 1:
        colors.append(red)
    else:
        colors.append(blue)
for idx, color in enumerate(colors):
    if color[0] == 0.8:
        x_s_red.append(x_s[idx])
        y_s_red.append(y_s[idx])
    else:
        x_s_blue.append(x_s[idx])
        y_s_blue.append(y_s[idx])
x = np.linspace(np.min(x_s), np.max(x_s), 100)
ax.plot(x, y_func(x), '-k', label='Decision Line')
plt.scatter(x_s_red, y_s_red, color = red, label = 'On')
plt.scatter(x_s_blue, y_s_blue, color = blue, label = 'Off')
plt.xlabel('X')
plt.ylabel('Y')
leg = ax.legend();
plt.show()

fig1, ax1 = plt.subplots()
a = [1,2,3,4,5,6]
b = [7.57,5.10,5.33,4.32,4.52,3.95]
plt.plot(a, b, '-k')
plt.xlabel('Number of Epochs')
plt.ylabel('Average Misclassification Rate (%)')
plt.show()



