import numpy as np

class Sigmoid:
    def g(Z):
        return 1/(1+np.exp(-Z))

    def del_g_z(a_in):
        return a_in * (1. - a_in)
    
    def del_J_z(a_in, y_actual, m):
        del_J_g = -((y_actual/a_in) - (1-y_actual)/(1-a_in))/m
        del_g_z = a_in * (1. - a_in)
        return del_J_g * del_g_z

class Linear:
    def g(Z):
        return Z
    
    def del_g_z(self):
        return 1
    
class Relu:
    def g(Z):
        output = np.zeros(Z.shape)
        for i in range(Z.shape[0]):
            output[i] = max(0., Z[i][0])
        return output
    
    def del_g_z(a_in):
        output = np.zeros(a_in.shape)
        for i in range(a_in.shape[0]):
            output[i] = 1 if a_in[i][0] > 0 else 0
        return output
    
class Costs:
    def sigmoid_cost(y_hat, y_actual):
        m = y_actual.shape[0]
        error = 0
        for i in range(m):
            error += (y_actual[i] * np.log(y_hat[i][0]) + (1 - y_actual[i]) * np.log(1 - y_hat[i][0]))
        return (-error/m).reshape(1)