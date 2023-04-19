import math

import matplotlib.pyplot as plt
import numpy as np

data_set_train = np.genfromtxt("hw05_data_set_train.csv", delimiter=",")
data_set_test = np.genfromtxt("hw05_data_set_test.csv", delimiter=",")

x_training = data_set_train[:, 0]
y_training = data_set_train[:, 1]

x_test = data_set_test[:, 0]
y_test = data_set_test[:, 1]

K = np.max(y_training)
N_training = x_training.shape[0]
N_test = x_test.shape[0]

min_val = 0
max_val = 2

# given in the pdf
P = 30


def safelog2(x):
    if x == 0:
        return (0)
    else:
        return (np.log2(x))


node_indices = {}
is_terminal = {}
need_split = {}
node_features = {}
node_splits = {}
node_indices[1] = np.array(range(N_training))
is_terminal[1] = False
need_split[1] = True


# define a decision tree function to get is_terminal, node_means, node_splits easily
# some part of the code taken from lab 7
def DecisionTree(P):
    while True:
        # find nodes that need splitting
        split_nodes = [key for key, value in need_split.items() if value == True]
        # check whether we reach all terminal nodes
        if len(split_nodes) == 0:
            break
        # find best split positions for all nodes
        for split_node in split_nodes:
            data_indices = node_indices[split_node]
            need_split[split_node] = False
            if data_indices.shape[0] <= P:
                is_terminal[split_node] = True
            else:
                is_terminal[split_node] = False
                best_scores = 0.0
                best_splits = 0.0
                unique_values = np.sort(np.unique(x_training[data_indices]))
                split_positions = (unique_values[1:len(unique_values)] + unique_values[0:(len(unique_values) - 1)]) / 2
                split_scores = np.repeat(0.0, len(split_positions))
                # no iteration for number of Feature(D)
                for s in range(len(split_positions)):
                    left = np.sum((x_training[data_indices] < split_positions[s]) * y_training[data_indices]) / np.sum(
                        (x_training[data_indices] < split_positions[s]))
                    right = np.sum(
                        (x_training[data_indices] >= split_positions[s]) * y_training[data_indices]) / np.sum(
                        (x_training[data_indices] >= split_positions[s]))
                    split_scores[s] = 1 / len(data_indices) * (np.sum(
                        (y_training[data_indices] - np.repeat(left, data_indices.shape[0], axis=0)) ** 2 * (
                                x_training[data_indices] < split_positions[s])) + np.sum(
                        (y_training[data_indices] - np.repeat(right, data_indices.shape[0], axis=0)) ** 2 * (
                                x_training[data_indices] >= split_positions[s])))

                best_scores = np.min(split_scores)
                best_splits = split_positions[np.argmin(split_scores)]

                node_splits[split_node] = best_splits
                # create left node using the selected split
                left_indices = data_indices[x_training[data_indices] < best_splits]
                node_indices[2 * split_node] = left_indices
                is_terminal[2 * split_node] = False
                need_split[2 * split_node] = True

                # create right node using the selected split
                right_indices = data_indices[x_training[data_indices] >= best_splits]
                node_indices[2 * split_node + 1] = right_indices
                is_terminal[2 * split_node + 1] = False
                need_split[2 * split_node + 1] = True


DecisionTree(P)

# resgersiion calculations for p = 30 for once
node_split_values = np.sort(np.array(list(node_splits.items()))[:, 1])
left_borders = np.append(min_val, np.transpose(np.sort(np.array(list(node_splits.items()))[:, 1])))
right_borders = np.append(np.transpose(np.sort(np.array(list(node_splits.items()))[:, 1])), max_val)

points = np.asarray([np.sum(((left_borders[b] < x_training) & (x_training <= right_borders[b])) * y_training) for b in
                     range(len(left_borders))]) / np.asarray(
    [np.sum((left_borders[b] < x_training) & (x_training <= right_borders[b])) for b in range(len(left_borders))])

plt.figure(figsize=(16, 6))
plt.plot(x_training, y_training, "b.", markersize=10, label="training")
for b in range(len(left_borders)):
    plt.plot([left_borders[b], right_borders[b]], [points[b], points[b]], "k-")
for b in range(len(left_borders) - 1):
    plt.plot([right_borders[b], right_borders[b]], [points[b], points[b + 1]], "k-")
plt.xlabel("Time (sec)")
plt.ylabel("Signal (millivolt)")
plt.legend(["training"], loc='upper right')
plt.show()

plt.figure(figsize=(16, 6))
plt.plot(x_test, y_test, "r.", markersize=10, label="test")
for b in range(len(left_borders)):
    plt.plot([left_borders[b], right_borders[b]], [points[b], points[b]], "k-")
for b in range(len(left_borders) - 1):
    plt.plot([right_borders[b], right_borders[b]], [points[b], points[b + 1]], "k-")
plt.xlabel("Time (sec)")
plt.ylabel("Signal (millivolt)")
plt.legend(["test"], loc='upper right')
plt.show()


def RMSE(N, x_t, y_t, points):
    error = 0
    for i in range(N):
        for a in range(len(left_borders)):
            if (left_borders[a] < x_t[i]) and (x_t[i] <= right_borders[a]):
                error += (y_t[i] - points[a]) ** 2
    return math.sqrt(error / N)


RMSE_training = RMSE(N_training, x_training, y_training, points)
RMSE_test = RMSE(N_test, x_test, y_test, points)

print("RMSE on training set is {} when P is {}".format(RMSE_training, P))
print("RMSE on test set is {} when P is {}".format(RMSE_test, P))

trainin_p_values = []
training_rmse_values = []
test_rmse_values = []

for p in range(10, 55, 5):
    node_indices = {}
    is_terminal = {}
    need_split = {}
    node_features = {}
    node_splits = {}
    node_indices[1] = np.array(range(N_training))
    is_terminal[1] = False
    need_split[1] = True
    DecisionTree(p)

    # resgersiion calculations for different p's
    node_split_values = np.sort(np.array(list(node_splits.items()))[:, 1])
    left_borders = np.append(min_val, np.transpose(np.sort(np.array(list(node_splits.items()))[:, 1])))
    right_borders = np.append(np.transpose(np.sort(np.array(list(node_splits.items()))[:, 1])), max_val)
    points = np.asarray([np.sum(((left_borders[b] < x_training) & (x_training
                                                                   <= right_borders[b])) * y_training) for b in
                         range(len(left_borders))]) / np.asarray(
        [np.sum((left_borders[b] < x_training) & (x_training <= right_borders[b])) for b in range(len(left_borders))])

    trainin_p_values.append(p)
    training_rmse_values.append(RMSE(N_training, x_training, y_training, points))
    test_rmse_values.append(RMSE(N_test, x_test, y_test, points))

plt.figure(figsize=(16, 6))
plt.plot(range(10, 55, 5), training_rmse_values, marker=".", markersize=10, linestyle="-", color="b", label='training')
plt.plot(range(10, 55, 5), test_rmse_values, marker=".", markersize=10, linestyle="-", color="r", label='test')
plt.xlabel("Pre-pruning size (P)")
plt.ylabel("RMSE")
plt.legend(['training', 'test'])
plt.show()
