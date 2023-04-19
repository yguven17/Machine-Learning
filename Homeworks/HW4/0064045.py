import math
import matplotlib.pyplot as plt
import numpy as np

data_set_train = np.genfromtxt("hw04_data_set_train.csv", delimiter=",")
data_set_test = np.genfromtxt("hw04_data_set_test.csv", delimiter=",")

x_training = data_set_train[:, 0]
y_training = data_set_train[:, 1]

x_test = data_set_test[:, 0]
y_test = data_set_test[:, 1]

K = np.max(y_training)
N = data_set_train.shape[0]

bin_width = 0.1
origin = 0.0

minimum_value = min(x_training)
maximum_value = max(x_training)
data_interval = np.linspace(minimum_value, maximum_value, 1601)

left_borders = np.arange(minimum_value, maximum_value - bin_width, bin_width)
right_borders = np.arange(minimum_value + bin_width, maximum_value, bin_width)

regressogram = np.zeros(len(left_borders))

for b in range(len(left_borders)):
    conditionalpart = ((left_borders[b] < x_training) & (x_training <= right_borders[b]))
    regressogram[b] = np.sum(conditionalpart * y_training) / np.sum(conditionalpart)

# plot1
plt.figure(figsize=(10, 6))
plt.plot(x_training, y_training, "b.", markersize=10, label="Training")

for b in range(len(left_borders)):
    plt.plot([left_borders[b], right_borders[b]], [regressogram[b], regressogram[b]], "k-")
for b in range(len(left_borders) - 1):
    plt.plot([right_borders[b], right_borders[b]], [regressogram[b], regressogram[b + 1]], "k-")

plt.ylabel("Signal (milivolt)")
plt.xlabel("Time (sec)")
plt.legend(loc='upper right')
plt.show()

# plot2
plt.figure(figsize=(10, 6))
plt.plot(x_test, y_test, "r.", markersize=10, label="Test")

for b in range(len(left_borders)):
    plt.plot([left_borders[b], right_borders[b]], [regressogram[b], regressogram[b]], "k-")
for b in range(len(left_borders) - 1):
    plt.plot([right_borders[b], right_borders[b]], [regressogram[b], regressogram[b + 1]], "k-")

plt.ylabel("Signal (milivolt)")
plt.xlabel("Time (sec)")
plt.legend(loc='upper right')
plt.show()

# rmse1 calculator
rmse1 = 0
for i in range(len(left_borders)):
    for j in range(len(y_test)):
        if right_borders[i] >= x_test[j] >= left_borders[i]:
            rmse1 += (y_test[j] - regressogram[i]) ** 2

rmse1 = math.sqrt(rmse1 / len(y_test))
print("Regressogram => RMSE is " + str(rmse1) + " when h is " + str(bin_width))

mean_smooth = np.asarray([np.sum((np.abs((x - x_training) / bin_width) <= 0.5) * y_training) / (
    np.sum((np.abs((x - x_training) / bin_width) <= 0.5))) for x in data_interval])

# plot3
plt.figure(figsize=(10, 6))
plt.plot(x_training, y_training, "b.", markersize=10, label="Training")
plt.plot(data_interval, mean_smooth, "k-")
plt.ylabel("Signal (milivolt)")
plt.xlabel("Time (sec)")
plt.legend(loc='upper right')
plt.show()

# plot4
plt.figure(figsize=(10, 6))
plt.plot(x_test, y_test, "r.", markersize=10, label="Test")
plt.plot(data_interval, mean_smooth, "k-")
plt.ylabel("Signal (milivolt)")
plt.xlabel("Time (sec)")
plt.legend(loc='upper right')
plt.show()


# rmse2 calculator

def easyCalculator1(u):
    return abs(u) <= 0.5

mean_smoother_rmse = np.array([np.sum(easyCalculator1((x - x_training) / bin_width) * y_training)
                               / np.sum(easyCalculator1((x - x_training) / bin_width)) for x in x_test])
rmse2 = np.sqrt(np.sum((y_test - mean_smoother_rmse) ** 2) / len(y_test))

print("Running Mean Smoother => RMSE is " + str(rmse2) + " when h is " + str(bin_width))


bin_width = 0.02

def easyCalculator2(u):
    return 1 / math.sqrt(2 * math.pi) * np.exp(-0.5 * (u ** 2))

kernel_smooth = np.array(
    [np.sum(easyCalculator2((x - x_training) / bin_width) * y_training) / np.sum(
        easyCalculator2((x - x_training) / bin_width)) for x in
     data_interval])

# plot5
plt.figure(figsize=(10, 6))
plt.plot(x_training, y_training, "b.", markersize=10, label="Training")
plt.ylabel("Signal (milivolt)")
plt.xlabel("Time (sec)")
plt.legend(loc='upper right')

plt.plot(data_interval, kernel_smooth, "k-")
plt.show()

# plot6
plt.figure(figsize=(10, 6))
plt.plot(x_test, y_test, "r.", markersize=10, label="Test")
plt.ylabel("Signal (milivolt)")
plt.xlabel("Time (sec)")
plt.legend(loc='upper right')

plt.plot(data_interval, kernel_smooth, "k-")
plt.show()

rmse3 = np.array(
    [np.sum(easyCalculator2((x - x_training) / bin_width) * y_training) / np.sum(
        easyCalculator2((x - x_training) / bin_width)) for x in x_test])
kernel_smoother_error = np.sqrt(np.sum((y_test - rmse3) ** 2) / len(y_test))
print("Kernel Smoother=> RMSE is " + str(kernel_smoother_error) + " when h is " + str(bin_width))
