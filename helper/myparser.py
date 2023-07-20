import csv
import numpy as np
import random

def parse(path, ignore_idx, params, total_count):
    curr_total = 0
    all_data = []

    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if curr_total >= total_count:
                return all_data
            else:
                curr_total += 1
            data_set = []
            for j in range(ignore_idx+1, ignore_idx+2+params):
                if row[j] == 'M':
                    data_set.append(1)
                elif row[j] == 'B':
                    data_set.append(0)
                else:
                    data_set.append(float(row[j]))
            all_data.append(data_set)
    
    return all_data

# using fisher yates algorithm
def scramble_data(all_data, seed=10):
    random.seed(seed)
    for i in range(len(all_data)-1, -1, -1):
        j = random.randint(0, i)
        temp = all_data[i]
        all_data[i] = all_data[j]
        all_data[j] = temp

def get_subarr(data, count):
    if count > len(data):
        raise Exception(f"Train count greater than size of data set by {count - len(data)}")
    
    X = []
    Y = []

    for i in range(count):
        X.append(data[0][1:])
        Y.append(data[0][0])
        data.pop(0)

    return np.array(X), np.array(Y)

def z_normalize(x_total):
    normalized_x_total = np.zeros(x_total.shape)
    
    n = x_total.shape[1]

    for j in range(n):
        mean_j = np.mean(x_total[:, j])
        std_j = np.std(x_total[:, j])
        new_x = np.array((x_total[:, j] - mean_j)/std_j)
        normalized_x_total[:, j] = new_x
        
    return normalized_x_total

def process_data(x_data, y_data):
    X_processed = []
    Y_processed = []
    for i in range(x_data.shape[0]):
        X_processed.append(x_data[i].reshape(x_data.shape[1], 1))
        Y_processed.append(y_data[i].reshape(1, 1))
    return np.array(X_processed), np.array(Y_processed)