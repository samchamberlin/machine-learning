import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split

from tools.arff import Arff
import matplotlib.pyplot as plt


class MLPClassifier(BaseEstimator,ClassifierMixin):

    def __init__(self,lr=.1, momentum=0, shuffle=False, hidden_layer_size=None, num_hidden_layers = 1):
        """ Initialize class with chosen hyperparameters.

        Args:
            lr (float): A learning rate / step size.
            shuffle(boolean): Whether to shuffle the training data each epoch. DO NOT SHUFFLE for evaluation / debug datasets.
            momentum(float): The momentum coefficent 
        Optional Args (Args we think will make your life easier):
            hidden_layer_size (list(int)): A list of integers which defines the width of each hidden layer if hidden layer is none do twice as many hidden nodes as input nodes.
        Example:
            mlp = MLPClassifier(lr=.2,momentum=.5,shuffle=False,hidden_layer_size = [3,3]),  <--- this will create a model with two hidden layers, both 3 nodes wide
        """
        self.lr = lr
        self.momentum = momentum
        self.shuffle = shuffle
        self.hidden_layer_size = hidden_layer_size
        self.num_hidden_layers = num_hidden_layers

        self.change_hidden_weights = 0
        self.change_weights = 0
        self.delta_weight = 0
        self.hidden_outputs = []
        self.accuracy_train = np.array([])
        self.accuracy_test = np.array([])
        self.accuracy_validation = np.array([])
        self.MSE_train = np.array([])
        self.MSE_validation = np.array([])
        self.MSE_test = np.array([])
        self.epochs_run = 0
        self.validation_score = 1

    def fit(self, p, t, initial_weights=[]):
        """ Fit the data; run the algorithm and adjust the weights to find a good solution
        Args:
            p (array-like): A 2D numpy array with the data, excluding targets
            t (array-like): A 2D numpy array with the targets
            initial_weights: A 2D array with the initial weights
        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)
        """

        # Split training and testing set and add bias
        self.split_data(p, t)

        # Initialize hidden layers
        if self.hidden_layer_size is None:
            # Default is to have double the number of input nodes as hidden nodes
            self.hidden_nodes = np.zeros((self.patterns_train.shape[1] - 1) * 2)
            # Use for testing
            # self.hidden_nodes = np.ones(self.patterns_train.shape[1])
        else:
            self.hidden_nodes = np.zeros(self.hidden_layer_size)

        # Initialize weights
        if initial_weights:
            self.weights = initial_weights[:, -self.targets_train.shape[1]]
            self.hidden_weights = initial_weights[:, self.targets_train.shape[1]:]
        else:
            self.initialize_weights()

        num_epochs_small_MSE_change = 0
        max_epochs_small_MSE_change = 20
        min_MSE = 1
        small_improvement = 0.003
        best_vs_MSE = 0

        while num_epochs_small_MSE_change < max_epochs_small_MSE_change:
        # while self.epochs_run < 10:
            if self.shuffle:
                self.shuffle_data()

            self.output_train = np.empty((0,t.shape[1]), float)

            # Update weights
            for i, pattern in enumerate(self.patterns_train):
                outputs = []
                hidden_outputs = []
                target_outputs = []
                target_deltas = []
                hidden_deltas = []
                change_hidden_weights = np.empty((0,2))
                change_weights = []
                sigmas = 0
                for j, weight_row in enumerate(self.weights):
                    net = np.dot(pattern, weight_row)
                    hidden_output = self.get_output(net)
                    # outputs.append(hidden_output)
                    hidden_outputs = np.append(hidden_outputs, hidden_output)
                # Add hidden bias output
                hidden_outputs = np.append(hidden_outputs, 1)
                if hidden_outputs.ndim == 1:
                    hidden_outputs = [hidden_outputs]
                change_hidden_weights = np.empty((0,self.hidden_weights.shape[1]), float)
                for j, hidden_weight_row in enumerate(self.hidden_weights):
                    net = np.dot(hidden_outputs, hidden_weight_row)
                    target_output = self.get_output(net)
                    target_outputs = np.append(target_outputs, target_output)
                    target_delta = self.get_delta_output_node(self.targets_train[i][j], target_output)
                    target_deltas = np.append(target_deltas, target_delta)
                    # Get hidden change in weights
                    change_hidden_weight = hidden_outputs[0] * target_delta * self.lr
                    change_hidden_weights = np.append(change_hidden_weights, np.array([change_hidden_weight]), axis=0)
                output_train = target_outputs
                # Backpropagation
                hidden_outputs = np.array(hidden_outputs)
                self.hidden_weights = np.array(self.hidden_weights)
                # Get hidden deltas
                for j,col in enumerate(self.hidden_weights.T):
                    # if hidden_outputs.shape[0] == 1:
                    hidden_delta = hidden_outputs[0][j] * (1 - hidden_outputs[0][j]) * np.dot(target_deltas, col)
                    # else:
                    #     hidden_delta = hidden_outputs[j] * (1 - hidden_outputs[j]) * np.dot(target_deltas, col)
                    hidden_deltas = np.append(hidden_deltas, hidden_delta)
                # Remove hidden bias delta
                hidden_deltas = hidden_deltas[:-1]

                # Get change in weights
                change_weights = np.empty((0, self.weights.shape[1]), float)
                for j, weight_row in enumerate(self.weights):
                    change_weight = pattern * hidden_deltas[j] * self.lr
                    change_weights = np.append(change_weights, np.array([change_weight]), axis=0)

                # Momentum term
                m_hidden = self.momentum * self.change_hidden_weights
                m_input = self.momentum * self.change_weights
                self.change_hidden_weights = change_hidden_weights
                self.change_weights = change_weights

                # Get new hidden weights
                self.hidden_weights += change_hidden_weights + m_hidden
                # Get new input weights
                self.weights += change_weights + m_input

                self.output_train = np.append(self.output_train, np.array([output_train]), axis=0)

            # Get Accuracy
            MSE_train, accuracy_train = self.score(self.patterns_train, self.targets_train, self.output_train)
            self.MSE_train = np.append(self.MSE_train, MSE_train)
            self.accuracy_train = np.append(self.accuracy_train, accuracy_train)

            output_test = self.predict(self.patterns_test)
            MSE_test, accuracy_test = self.score(self.patterns_test, self.targets_test, output_test)
            self.MSE_test = np.append(self.MSE_test, MSE_test)

            output_validation = self.predict(self.patterns_validation)
            MSE_validation, accuracy_validation = self.score(self.patterns_validation, self.targets_validation, output_validation)
            self.MSE_validation = np.append(self.MSE_validation, MSE_validation)
            if self.accuracy_validation.size != 0:
                if self.accuracy_validation[-1] < accuracy_validation:
                    self.accuracy_validation = np.append(self.accuracy_validation, accuracy_validation)
                else:
                    self.accuracy_validation = np.append(self.accuracy_validation, self.accuracy_validation[-1])
            else:
                self.accuracy_validation = np.append(self.accuracy_validation, accuracy_validation)

            if np.abs(MSE_validation - min_MSE) < small_improvement:
                num_epochs_small_MSE_change += 1
            else:
                num_epochs_small_MSE_change = 0

            if MSE_validation < min_MSE:
                min_MSE = MSE_validation

            self.epochs_run += 1
        #     if MSE_validation < best_vs_MSE:
        #         best_vs_MSE = MSE_validation
        #
        # x = np.where(self.MSE_validation == best_vs_MSE)
        # self.MSE_validation = self.MSE_validation[:x]

        return self

    def predict(self, p):
        """ Predict all classes for a dataset X
        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
        Returns:
            array, shape (n_samples,)
                Predicted target values per element in X.
        """
        o = np.empty((0,self.targets_validation.shape[1]), float)
        for i, pattern in enumerate(p):
            hidden_outputs = []
            target_outputs = []
            for j, weight_row in enumerate(self.weights):
                net = np.dot(pattern, weight_row)
                hidden_output = self.get_output(net)
                hidden_outputs = np.append(hidden_outputs, hidden_output)
            hidden_outputs = np.append(hidden_outputs, 1)
            for j, hidden_weight_row in enumerate(self.hidden_weights):
                net = np.dot(np.array(hidden_outputs), hidden_weight_row)
                target_output = self.get_output(net)
                target_outputs = np.append(target_outputs, target_output)
            o = np.append(o, np.array([target_outputs]), axis=0)

        return o

    def initialize_weights(self):
        """ Initialize weights for perceptron.
        """

        num_inputs = self.patterns_train.shape[1]
        num_hidden_nodes = self.hidden_nodes.size
        num_targets = self.targets_train.shape[1]

        # Random weights
        self.weights = np.random.normal(0, 0.25, size = (num_hidden_nodes, num_inputs))
        self.hidden_weights = np.random.normal(0, 0.25, size = (num_targets, num_hidden_nodes + 1))

        # All zero weights
        # self.weights = np.zeros((num_hidden_nodes, num_inputs))
        # self.hidden_weights = np.zeros((num_targets, num_hidden_nodes + 1))

        # All one weights
        # self.weights = np.ones((num_hidden_nodes, num_inputs))
        # self.hidden_weights = np.ones((num_targets, num_hidden_nodes + 1))

        pass

    def score(self, p, t, o):
        """ Return accuracy of model on a given dataset. Must implement own score function.
        """
        # ========== MSE ==========
        MSE = np.sum((t - o)**2) / t.shape[0]
        # =========================

        # ========== ACCURACY ==========
        accuracy = 0
        for idx,target in enumerate(t):
            # Flatten outputs
            o_flat = o[idx]
            for i,val in enumerate(o_flat):
                max = np.amax(o_flat)
                if o_flat[i] == max:
                    o_flat[i] = 1
                else:
                    o_flat[i] = 0

        # Check targets vs flat outputs
            A = o_flat
            if np.array_equal(target, A):
                accuracy += 1

        accuracy = (accuracy / t.size)
        # ==============================

        return MSE, accuracy

    def _shuffle_data(self):
        """ Shuffle the data!
        """
        data = np.column_stack((self.patterns_train, self.targets_train))
        np.random.shuffle(data)
        self.patterns_train = data[:, 0:-1]
        self.targets_train = data[:, -1]

        pass

    def split_data(self, p, t):
        # Splitting Data
        self.p_train, self.patterns_test, self.t_train, self.targets_test = \
            train_test_split(p, t, test_size=0.25, random_state=42)
        self.patterns_train, self.patterns_validation, self.targets_train, self.targets_validation = \
            train_test_split(self.p_train, self.t_train, test_size=.15, random_state=42)

        if len(t.shape) == 1:
            self.targets_train = np.array([self.targets_train]).reshape(-1, 1)
            self.targets_test = np.array([self.targets_test]).reshape(-1, 1)

        # If not splitting data
        # self.patterns_train = p
        # self.patterns_test = p
        #
        # if len(t.shape) == 1:
        #     self.targets_train = np.array([t]).reshape(-1, 1)
        #     self.targets_test = np.array([t]).reshape(-1, 1)
        # else:
        #     self.targets_train = t
        #     self.targets_test = t

        # Add bias
        self.patterns_train = np.append(self.patterns_train, np.ones(self.patterns_train.shape[0]).reshape(-1, 1),
                                        axis=1)
        self.patterns_test = np.append(self.patterns_test, np.ones(self.patterns_test.shape[0]).reshape(-1, 1),
                                        axis=1)
        self.patterns_validation = np.append(self.patterns_validation, np.ones(self.patterns_validation.shape[0]).reshape(-1, 1),
                                        axis=1)
        pass

    def get_weights(self):
        weights = np.append(self.hidden_weights.reshape(1, -1) , self.weights.reshape(1, -1))
        return weights

    def get_output(self, net):
        return 1/(1+np.exp(-net))

    def get_delta_output_node(self, target, output):
        return (target - output) * output * (1 - output)

    def get_delta_hidden_nodes(self, hidden_outputs, delta):
        # TODO: not sure if this is right
        return np.sum(delta * self.hidden_weights) * hidden_outputs * (1 - hidden_outputs)

    def get_weight_change(self, output, delta):
        return self.C * output * delta


# Data
data = Arff("../data/backpropagation/iris.arff", label_count=1)
# data = Arff("../data/backpropagation/vowels.arff", label_count=1)
# data = Arff("../data/backpropagation/linsep2nonorigin.arff", label_count=1)
# data = Arff("../data/backpropagation/data_banknote_authentication.arff", label_count=1)
# data = Arff("../data/backpropagation/class_ex.arff", label_count=1)

patterns = data[:, 0:-1]
targets = data[:, [-1]]

# ========== For iris data set ==========
new_targets = []
for idx,t in enumerate(targets):
    if t == 0:
        # targets[idx] = [1,0,0]
        new_targets.append([1, 0, 0])
    if t == 1:
        # targets[idx] = [0, 1, 0]
        new_targets.append([0, 1, 0])
    if t == 2:
        # targets[idx] = [0, 0, 1]
        new_targets.append([0, 0, 1])
targets = np.array(new_targets)
# =======================================

# ========== Normalize Data ==========
mean = np.mean(patterns, axis=0)
var = np.var(patterns, axis=0)

patterns = (patterns - mean) / var
# ====================================


# Test
mlp = MLPClassifier(lr=0.1, momentum=.5, shuffle=False, num_hidden_layers=1)
mlp.fit(patterns, targets)
print(mlp.get_weights())

A = mlp.MSE_train
x1 = np.arange(A.shape[0]) + 1
y1 = A

B = mlp.MSE_validation
x2 = np.arange(B.shape[0]) + 1
y2 = B

C = mlp.accuracy_validation
x3 = np.arange(C.shape[0]) + 1
y3 = C

# MSE vs epochs
# plt.plot(x1,y1, '-b', label='Training Set')
# plt.plot(x2,y2, '-r', label='Validation Set')
# plt.xlabel('Number of Epochs')
# plt.ylabel('MSE')
# plt.legend(loc="upper right")
# plt.show()

# Accuracy vs epochs
plt.plot(x3,y3, '-r', label='Validation Set')
plt.xlabel('Number of Epochs')
plt.ylabel('Accuracy (%)')
plt.legend(loc="upper right")
plt.show()
