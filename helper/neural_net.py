import numpy as np
from helper.activations import Sigmoid, Linear, Relu, Costs

class Layer:
    def __init__(self, units, activation):
        self.activation = activation
        self.units = units
        self.neurons = np.array([])
        self.W_l = np.array([])
        self.B_l = np.array([])
    
    def initialize(self, prev_layer_units, seed=10):
        np.random.seed(seed)
        W_shape = (self.units, prev_layer_units)
        B_shape = (self.units, 1)
        self.W_l = np.random.randn(W_shape[0], W_shape[1]) * np.sqrt(2/W_shape[1])
        self.B_l = np.zeros(B_shape)

    # feeds data through single layer
    def feed_forward(self, a_in):
        Z = np.matmul(self.W_l, a_in) + self.B_l
        self.neurons = self.activation.g(Z)
    
    # backprops weights in single layer
    def back_prop(self, del_J_z, prev_layer, alpha, reg_rate, m):
        del_J_w = np.matmul(prev_layer.neurons, del_J_z.T)
        del_J_b = del_J_z

        prev_del_J_z = np.multiply(np.matmul(self.W_l.T, del_J_z), prev_layer.activation.del_g_z(prev_layer.neurons))

        self.W_l -= alpha * del_J_w.T + (alpha * reg_rate * self.W_l)/m
        self.B_l -= alpha * del_J_b

        return prev_del_J_z

    def summarize(self):
        print("Weights:")
        print(self.W_l)
        print('\n')
        print("Biases:")
        print(self.B_l)
        print('\n')
        print("Activations:")
        print(self.neurons)

class Model:
    def __init__(self, layer_arr):
        self.layers = np.array(layer_arr) 

    def initialize(self, n, seed=10):
        np.random.seed(seed)
        for l in range(self.layers.shape[0]-1, -1, -1):
            seed_l = np.random.randint(0, 100)
            prev_units = n if l == 0 else self.layers[l-1].units
            self.layers[l].initialize(prev_units, seed_l)

    # forward propagate through entire model
    def forward_propagate(self, x_in):
        a_in = x_in
        for l in range(self.layers.shape[0]):
            self.layers[l].feed_forward(a_in)
            a_in = self.layers[l].neurons
        return a_in

    # back propagates through entire model
    def back_propagate(self, m, x_in, y_out, alpha, reg_rate):
        a_out = self.layers[-1].neurons
        del_J_z = self.layers[-1].activation.del_J_z(a_out, y_out, m)

        x_in_layer = Layer(x_in.shape[0], Linear)
        x_in_layer.neurons = x_in

        for l in range(self.layers.shape[0]-1, -1, -1):
            prev_layer = x_in_layer if l == 0 else self.layers[l-1]
            del_J_z = self.layers[l].back_prop(del_J_z, prev_layer, alpha, reg_rate, m)

    def fit(self, X_train, Y_train, X_test, Y_test, alpha, epochs, seed=10, reg_rate=0):
        n = X_train.shape[1]
        self.initialize(n, seed)
    
        train_J_hist = []
        train_acc_hist = []
        
        test_J_hist = []
        test_acc_hist = []

        for c in range(epochs):
            test_results = self.test(X_test, Y_test, reg_rate)

            test_J_hist.append(test_results[0])
            test_acc_hist.append(test_results[1])

            train_results = self.train(X_train, Y_train, alpha, reg_rate)

            train_J_hist.append(train_results[0])
            train_acc_hist.append(train_results[1])

            print(f"Epoch {c+1} complete!")
        
        return train_J_hist, test_J_hist, train_acc_hist, test_acc_hist
    
    def test(self, X_test, Y_test, reg_rate):
        test_accuracy = 0

        test_preds = []

        for i in range(X_test.shape[0]):
            pred = self.forward_propagate(X_test[i])
            test_preds.append(pred)

            if (pred > 0.5 and Y_test[i][0] == 1) or (pred < 0.5 and Y_test[i][0] == 0):
                test_accuracy += 1
        
        test_cost = Costs.sigmoid_cost(test_preds, Y_test) + self.get_l2_reg(reg_rate, X_test.shape[0])
        
        return [test_cost, test_accuracy/X_test.shape[0]]
    
    def train(self, X_train, Y_train, alpha, reg_rate):
        train_accuracy = 0

        train_preds = []

        m = X_train.shape[0]

        for i in range(X_train.shape[0]):
            pred = self.forward_propagate(X_train[i])
            train_preds.append(pred)

            if (pred > 0.5 and Y_train[i][0] == 1) or (pred < 0.5 and Y_train[i][0] == 0):
                    train_accuracy += 1

            self.back_propagate(m, X_train[i], Y_train[i], alpha, reg_rate)
        
        train_cost = Costs.sigmoid_cost(train_preds, Y_train) + self.get_l2_reg(reg_rate, m)

        return [train_cost, train_accuracy/m]

    def get_l2_reg(self, reg_rate, m):
        sum = 0
        for layer in self.layers:
            for i in range(layer.W_l.shape[0]):
                for j in range(layer.W_l.shape[1]):
                    sum += (layer.W_l[i][j] ** 2)
        return (sum * reg_rate) / (2 * m)

    def summarize(self):
        c = 0
        for layer in self.layers:
            print(f"Layer {c+1}")
            print("---------------------------")
            layer.summarize()
            print("---------------------------")
            print('\n')
            c += 1