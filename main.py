import numpy as np
import matplotlib.pyplot as plt
import myparser as parser
from neural_net import Model, Layer
from activations import Sigmoid, Relu


path = "/Users/benryben/Desktop/AI Self Learning/advanced_learning/neural_networks/projects/cancer_detection/data.csv"

test_count = 100

raw_X_train, raw_Y_train, raw_X_test, raw_Y_test = parser.parse(path=path, x_final_idx=9, test_count=test_count, total_count=569)

raw_X_train = np.array(raw_X_train)
raw_Y_train = np.array(raw_Y_train)

raw_X_test = np.array(raw_X_test)
raw_Y_test = np.array(raw_Y_test)

normalized_X_train = parser.z_normalize(raw_X_train)
normalized_X_test = parser.z_normalize(raw_X_test)

X_train, Y_train = parser.process_data(normalized_X_train, raw_Y_train)
X_test, Y_test = parser.process_data(normalized_X_test, raw_Y_test)

model_1 = Model([Layer(3, Relu), Layer(2, Relu), Layer(1, Sigmoid)])

alpha = 1.5
epochs = 200
seed = 5

# Train and test data
train_hist, test_hist, train_acc_hist, test_acc_hist = model_1.fit(X_train, Y_train, X_test, Y_test, alpha, epochs, seed)

# Print Results
print(f"Training error: {train_hist[epochs-1]}")
print(f"Test error: {test_hist[epochs-1]}")
print(f"Train accuracy: {train_acc_hist[epochs-1]}")
print(f"Test accuracy: {test_acc_hist[epochs-1]}")

plt.subplot(2, 1, 1)
plt.title("Learning curves")
plt.plot(np.arange(epochs), train_hist)
plt.plot(np.arange(epochs), test_hist)

plt.subplot(2, 1, 2)
plt.title("Accuracy")
plt.plot(np.arange(epochs), train_acc_hist)
plt.plot(np.arange(epochs), test_acc_hist)

plt.show()