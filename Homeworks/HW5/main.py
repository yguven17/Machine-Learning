import math
import matplotlib.pyplot as plt
import numpy as np

def safelog2(x):
    if x == 0:
        return(0)
    else:
        return(np.log2(x))


data_train = np.genfromtxt("hw05_data_set_train.csv", delimiter = ",")
data_test = np.genfromtxt("hw05_data_set_test.csv", delimiter = ",")

x_train = data_train[:,0]
x_test = data_test[:,0]

y_train = data_train[:,1]
y_test = data_test[:,1]

N_train = x_train.shape[0]
N_test = x_test.shape[0]


node_indices = {}
is_terminal = {}
need_split = {}
node_features = {} 
node_splits = {}
node_indices[1] = np.array(range(N_train))
is_terminal[1] = False
need_split[1] = True


P=30
minval = 0
maxval = 2
def Tree(prePrunningParam):
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
            if data_indices.shape[0] <= prePrunningParam:
                is_terminal[split_node] = True
            else:
                is_terminal[split_node] = False
                best_scores = 0.0
                best_splits = 0.0
                unique_values = np.sort(np.unique(x_train[data_indices]))
                split_positions = (unique_values[1:len(unique_values)] + unique_values[0:(len(unique_values) - 1)]) / 2
                split_scores = np.repeat(0.0, len(split_positions))
                # no iteration for number of Feature(D)
                for s in range(len(split_positions)):
                    left = np.sum((x_train[data_indices] < split_positions[s]) * y_train[data_indices]) / np.sum((x_train[data_indices] < split_positions[s]))
                    right = np.sum((x_train[data_indices] >= split_positions[s]) * y_train[data_indices]) / np.sum((x_train[data_indices] >= split_positions[s]))
                    split_scores[s] = 1 / len(data_indices) * (np.sum((y_train[data_indices]-np.repeat(left, data_indices.shape[0], axis = 0))**2 * (x_train[data_indices] < split_positions[s])) + np.sum((y_train[data_indices] - np.repeat(right, data_indices.shape[0], axis = 0))**2 * (x_train[data_indices] >= split_positions[s])))
                
                best_scores = np.min(split_scores)
                best_splits = split_positions[np.argmin(split_scores)]

                node_splits[split_node] = best_splits
                # create left node using the selected split
                left_indices = data_indices[x_train[data_indices] < best_splits]
                node_indices[2 * split_node] = left_indices
                is_terminal[2 * split_node] = False
                need_split[2 * split_node] = True

                # create right node using the selected split
                right_indices = data_indices[x_train[data_indices] >= best_splits]
                node_indices[2 * split_node + 1] = right_indices
                is_terminal[2 * split_node + 1] = False
                need_split[2 * split_node + 1] = True
                
                #print(node_splits)




Tree(P)
node_split_values = np.sort(np.array(list(node_splits.items()))[:,1])
left_borders = np.append(minval, np.transpose(np.sort(np.array(list(node_splits.items()))[:,1])))
right_borders = np.append(np.transpose(np.sort(np.array(list(node_splits.items()))[:,1])), maxval)

p_hat = np.asarray([np.sum(((left_borders[b] < x_train) & (x_train <= right_borders[b])) * y_train) for b in range(len(left_borders))]) / np.asarray([np.sum((left_borders[b] < x_train) & (x_train <= right_borders[b])) for b in range(len(left_borders))])

plt.figure(figsize = (15, 6))
plt.ylabel("Signal (milivolt)")
plt.xlabel("Time (sec)")
plt.plot(x_train, y_train, "b.", label = 'training', markersize = 10)
plt.legend(loc='upper right')
for b in range(len(left_borders)):
    plt.plot([left_borders[b], right_borders[b]], [p_hat[b], p_hat[b]], "k-")
for b in range(len(left_borders) - 1):
    plt.plot([right_borders[b], right_borders[b]], [p_hat[b], p_hat[b + 1]], "k-")
plt.show()

plt.figure(figsize = (15, 6))
plt.ylabel("Signal (milivolt)")
plt.xlabel("Time (sec)")
plt.plot(x_test, y_test, "r.", label = 'test', markersize = 10)
plt.legend(loc='upper right')
for b in range(len(left_borders)):
    plt.plot([left_borders[b], right_borders[b]], [p_hat[b], p_hat[b]], "k-")
for b in range(len(left_borders) - 1):
    plt.plot([right_borders[b], right_borders[b]], [p_hat[b], p_hat[b + 1]], "k-")
plt.show()




def RMSE_error(N,x,y,p_hat):    
    sum_err = 0
    for i in range(N):
        for a in range(len(left_borders)):
            if (left_borders[a] < x[i]) and (x[i] <= right_borders[a]):
                sum_err += (y[i] - p_hat[a])**2
    return math.sqrt(sum_err/N)


RMSE_test = RMSE_error(N_train,x_train,y_train,p_hat)
RMSE_training = RMSE_error(N_test,x_test,y_test,p_hat)


print("RMSE on training set is", RMSE_test, "when P is 30")
print("RMSE on test set is", RMSE_training, "when P is 30")



P_values_train = []
P_values_test = []
rmse_values_train = []
rmse_values = []
for p in range(10,55,5):
    node_indices = {}
    is_terminal = {}
    need_split = {}
    node_features = {} 
    node_splits = {}
    node_indices[1] = np.array(range(N_train))
    is_terminal[1] = False
    need_split[1] = True
    Tree(p)
    node_split_values = np.sort(np.array(list(node_splits.items()))[:,1])
    left_borders = np.append(minval, np.transpose(np.sort(np.array(list(node_splits.items()))[:,1])))
    right_borders = np.append(np.transpose(np.sort(np.array(list(node_splits.items()))[:,1])), maxval)
    p_hat = np.asarray([np.sum(((left_borders[b] < x_train) & (x_train <= right_borders[b])) * y_train) for b in range(len(left_borders))]) / np.asarray([np.sum((left_borders[b] < x_train) & (x_train <= right_borders[b])) for b in range(len(left_borders))])
    rmse_regress_train =RMSE_error(N_train,x_train,y_train,p_hat)
    P_values_train.append(p)
    rmse_values_train.append(rmse_regress_train)
    rmse_regress = RMSE_error(N_test,x_test,y_test,p_hat)
    rmse_values.append(rmse_regress)


plt.figure(figsize = (15, 5))
plt.plot(range(10,55,5), rmse_values_train, marker = ".", markersize = 10, linestyle = "-", color = "b",label='train')
plt.plot(range(10,55,5), rmse_values, marker = ".", markersize = 10, linestyle = "-", color = "r",label='test')
plt.xlabel("Pre-pruning size (P)")
plt.ylabel("RMSE")
plt.legend(['training', 'test'])
plt.show()







