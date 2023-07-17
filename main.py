import numpy as np
import matplotlib.pyplot as plt
import myparser as parser
from neural_net import Model, Layer
from activations import Sigmoid, Relu


path = "/Users/benryben/Desktop/AI Self Learning/advanced_learning/neural_networks/projects/cancer_detection/data.csv"

test_count = 100

raw_X_train, raw_Y_train, raw_X_test, raw_Y_test = parser.parse(path=path, x_final_idx=4, test_count=test_count)

raw_X_train = np.array(raw_X_train)
raw_Y_train = np.array(raw_Y_train)

raw_X_test = np.array(raw_X_test)
raw_Y_test = np.array(raw_Y_test)

normalized_X_train = parser.mean_normalize(raw_X_train)
normalized_X_test = parser.mean_normalize(raw_X_test)

X_train, Y_train = parser.process_data(normalized_X_train, raw_Y_train)
X_test, Y_test = parser.process_data(normalized_X_test, raw_Y_test)

model_1 = Model([Layer(3, Relu), Layer(2, Relu), Layer(1, Sigmoid)])

alpha = 11.5
epochs = 200
seed = 1

train_hist = model_1.fit(X_train, Y_train, alpha, epochs, seed)

print(train_hist[epochs-1])

plt.plot(np.arange(epochs), train_hist)
plt.show()
