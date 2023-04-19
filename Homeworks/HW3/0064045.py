import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# create array for usages

images_training_set = []
label_training_set = []
images_test_set = []
label_test_set = []
objective_function_values = []

# read data from cs files
images_set = np.genfromtxt("hw03_data_set_images.csv", delimiter=",")
labels_set = (np.genfromtxt("hw03_data_set_labels.csv", delimiter=",", dtype=str)).astype(str)

# find k and n from sets
K = 5
N = images_set.shape[0]
classes = np.unique(labels_set)

for i in range(N):
    result = np.argwhere(classes == labels_set[i])
    labels_set[i] = result[0][0]

labels_set = labels_set.astype(int)

# separate test and training data
for i in range(0, 5):
    images_training_set[(39 * i):(25 + 39 * i)] = images_set[(39 * i):(25 + 39 * i)]
    label_training_set[(25 + i * 39):39 * (i + 1)] = images_set[(25 + i * 39):39 * (i + 1)]
    images_test_set[(39 * i):(25 + 39 * i)] = labels_set[(39 * i):(25 + 39 * i)]
    label_test_set[(25 + i * 39):39 * (i + 1)] = labels_set[(25 + i * 39):39 * (i + 1)]

images_training_set = np.array(images_training_set)
label_training_set = np.array(label_training_set)
images_test_set = np.array(images_test_set)
label_test_set = np.array(label_test_set)

# print(images_training_set)
# print()
# print(label_training_set)
# print()
# print(images_test_set)
# print()
# print(label_test_set)
# print()

# create truth values and take them from set
images_test_set_truth = np.zeros((len(images_training_set), K)).astype(int)

for i in range(len(images_training_set)):
    images_test_set_truth[i][images_test_set[i]] = 1


# print(images_test_set_truth)

# define w function
def wFunction(xValue, truth, predicted):
    return (np.asarray(
        [np.matmul((truth[:, c] - predicted[:, c]) * (predicted[:, c]) * (predicted[:, c] - 1), xValue) for c in
         range(K)]).transpose())


# define w0 function
def w0Function(truth, predicted):
    return np.sum((truth - predicted) * predicted * (predicted - 1), axis=0)


# define sigmoid function
def sigmoidFunction(xValue, wValue, w0Value):
    return 1 / (1 + np.exp(-(np.matmul(wValue.T, xValue.T) + w0Value.T)))


np.random.seed(521)
w = np.random.uniform(low=-0.01, high=0.01, size=(images_training_set.shape[1], K))
w0 = np.random.uniform(low=-0.01, high=0.01, size=(1, K))

# given eta and epsilon values
eta = 0.001
epsilon = 0.001


# using while calculate w and w0
iteration = 1
while 1:
    images_training_set_predicted = sigmoidFunction(images_training_set, w, w0)
    objective_function_values = np.append(objective_function_values,
                                          0.5 * np.sum((images_test_set_truth - images_training_set_predicted.T) ** 2))
# keep old values for if check
    w0_prev = w0
    w_prev = w

# update w and w0
    w = w - eta * wFunction(images_training_set, images_test_set_truth, images_training_set_predicted.T)
    w0 = w0 - eta * w0Function(images_test_set_truth, images_training_set_predicted.T)

# checks if value grater than epsilon break the while for true calculations and plot
    if np.sqrt(np.sum((w0 - w0_prev) ** 2) + np.sum((w - w_prev) ** 2)) < epsilon:
        break

    iteration = iteration + 1

print("print(w)")
print(w)
print()
print("print(w0)")
print(w0)
print()

# crate figure for plot
plt.figure(figsize=(10, 6))
plt.plot(range(1, iteration + 1), objective_function_values, "k-")
plt.xlabel("Iteration")
plt.ylabel("Error")
plt.show()

# calculate confusion matrix
images_training_set_predicted = np.argmax(images_training_set_predicted, axis=0) + 1
confusion_matrix = pd.crosstab(images_training_set_predicted, images_test_set + 1, rownames=['y_pred'],
                               colnames=['y_truth'])
print("print(confusion_matrix)")
print(confusion_matrix)
print()

# calculate confusion matrix
label_training_set_predicted = sigmoidFunction(label_training_set, w, w0)
label_training_set_predicted = np.argmax(label_training_set_predicted, axis=0) + 1
confusion_matrix = pd.crosstab(label_training_set_predicted, label_test_set + 1, rownames=['y_pred'],
                               colnames=['y_truth'])
print("print(confusion_matrix)")
print(confusion_matrix)
