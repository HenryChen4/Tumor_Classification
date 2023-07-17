import csv
import numpy as np

def parse(path, x_final_idx, test_count):
    x_train_arr = []
    y_train_arr = []

    x_test_arr = []
    y_test_arr = []
    
    i = 0
    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            x_train_features = []
            x_test_features = []

            c = 0
            for item in row:
                if i < test_count:
                    try:
                        float_item = float(item)
                        if c > 1 and c < x_final_idx:
                            x_test_features.append(float_item)
                        c += 1
                    except:
                        if c == 1:
                            if item == 'M':
                                y_test_arr.append([1])
                            else:
                                y_test_arr.append([0])
                        c += 1
                        continue                    
                else:
                    try:
                        float_item = float(item)
                        if c > 1 and c < x_final_idx:
                            x_train_features.append(float_item)
                        c += 1
                    except:
                        if c == 1:
                            if item == 'M':
                                y_train_arr.append([1])
                            else:
                                y_train_arr.append([0])
                        c += 1
                        continue
                        
            if i < test_count:
                x_test_arr.append(x_test_features)
            else:
                x_train_arr.append(x_train_features)
            
            i += 1
    
    return x_train_arr, y_train_arr, x_test_arr, y_test_arr

def z_normalize(x_total):
    normalized_x_total = np.zeros(x_total.shape)
    
    m = x_total.shape[0]
    n = x_total.shape[1]

    mean = 0

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
        Y_processed.append(y_data[i].reshape(y_data.shape[1], 1))
    return np.array(X_processed), np.array(Y_processed)