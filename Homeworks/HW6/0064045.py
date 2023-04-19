import math
import matplotlib.pyplot as plt
import numpy as np
import cvxopt as cvx
import pandas as pd
import scipy.spatial.distance as dt

images_data = np.genfromtxt("hw06_data_set_images.csv", delimiter=",")
labels_data = np.genfromtxt("hw06_data_set_labels.csv", delimiter=",")

training_images = images_data[:1000, :]
test_images = images_data[1000:, :]
training_label = labels_data[:1000]
test_label = labels_data[1000:]

X_training = training_images
y_training = training_label.astype(int)
X_test = test_images
y_test = test_label.astype(int)

N_training = X_training.shape[0]
D_training = X_training.shape[1]
N_test = X_test.shape[0]
K = int(np.max(training_label))


def gaussian_kernel(X1, X2, s):
    D = dt.cdist(X1, X2)
    K = np.exp(-D ** 2 / (2 * s ** 2))
    return (K)

H_training=[]
for i in range (1000):
    hist, bin_edges = np.histogram((X_training[i]), bins=64);
    H_training.append((hist/784))
H_training = np.array(H_training)
print("H_training")
print(H_training[0:5,0:5])


H_test=[]
for i in range (1000):
    hist, bin_edges = np.histogram((X_test[i]), bins=64);
    H_test.append((hist/784))
H_test = np.array(H_test)
print("H_test")
print(H_test[0:5,0:5])

s = 10
K_training = gaussian_kernel(X_training, X_training, s)
K_test = gaussian_kernel(X_test, X_training, s)
print(K_training[0:5, 0:5])
print(K_test[0:5, 0:5])

epsilon = 1e-3


def OVA(C, s=10):
    f_predicteds = []
    f_predictedsTest = []
    for i in range(1, K + 1):
        y_trainingNew = np.copy(y_training)
        y_trainingNew[y_trainingNew != i] = -1.0
        y_trainingNew[y_trainingNew == i] = 1.0

        yyK = np.matmul(y_trainingNew[:, None], y_trainingNew[None, :]) * K_training

        P = cvx.matrix(yyK)
        q = cvx.matrix(-np.ones((N_training, 1)))
        G = cvx.matrix(np.vstack((-np.eye(N_training), np.eye(N_training))))
        h = cvx.matrix(np.vstack((np.zeros((N_training, 1)), C * np.ones((N_training, 1)))))
        A = cvx.matrix(1.0 * y_trainingNew[None, :])
        b = cvx.matrix(0.0)

        # use cvxopt library to solve QP problems
        result = cvx.solvers.qp(P, q, G, h, A, b)
        alpha = np.reshape(result["x"], N_training)
        alpha[alpha < C * epsilon] = 0
        alpha[alpha > C * (1 - epsilon)] = C

        # find bias parameter
        support_indices, = np.where(alpha != 0)
        active_indices, = np.where(np.logical_and(alpha != 0, alpha < C))
        w0 = np.mean(y_trainingNew[active_indices] * (
                1 - np.matmul(yyK[np.ix_(active_indices, support_indices)], alpha[support_indices])))

        # calculate predictions on traininging samples
        f_predicted = np.matmul(K_training, y_trainingNew[:, None] * alpha[:, None]) + w0
        f_predicteds.append(f_predicted)

        # calculate predictions on traininging samples
        f_predictedTest = np.matmul(K_test, y_trainingNew[:, None] * alpha[:, None]) + w0
        f_predictedsTest.append(f_predictedTest)

    return f_predicteds, f_predictedsTest


C = 10
f_predicteds, f_predictedsTest = OVA(C)

Y_predicted = (np.reshape(np.stack(np.transpose(f_predicteds), axis=-1), (1000, 1)))
y_predicted = np.argmax(Y_predicted, axis=1) + 1
confusion_matrix = pd.crosstab(np.reshape(y_predicted, N_training), y_training, rownames=['y_predicted'],
                               colnames=['y_training'])
print(confusion_matrix)

Y_predictedTest = (np.reshape(np.stack(np.transpose(f_predictedsTest), axis=-1), (1000, 1)))
y_predictedTest = np.argmax(Y_predictedTest, axis=1) + 1
confusion_matrix2 = pd.crosstab(np.reshape(y_predictedTest, N_test), y_test, rownames=['y_predicted'],
                                colnames=['y_test'])
print(confusion_matrix2)


Cs = [10 ** (-1), 10 ** (-0.5), 10 ** (0), 10 ** (0.5), 10 ** (1), 10 ** (1.5), 10 ** (2), 10 ** (2.5), 10 ** (3)]
accuracyTrains = []
accuracyTests = []
for C in Cs:
    f_predicteds, f_predictedsTest = OVA(C)
    Y_predicted = (np.reshape(np.stack(np.transpose(f_predicteds), axis=-1), (1000, 1)))
    y_predicted = np.argmax(Y_predicted, axis=1) + 1
    Y_predictedTest = (np.reshape(np.stack(np.transpose(f_predictedsTest), axis=-1), (1000, 1)))
    y_predictedTest = np.argmax(Y_predictedTest, axis=1) + 1

    accuracyTraining = 0
    for i in range(len(y_predicted)):
        if y_predicted[i] == y_training[i]:
            accuracyTraining = accuracyTraining + 1
    accuracyTraining = accuracyTraining / N_training
    accuracyTrains.append(accuracyTraining)

    accuracyTest = 0
    for i in range(len(y_predictedTest)):
        if y_predictedTest[i] == y_test[i]:
            accuracyTest = accuracyTest + 1
    accuracyTest = accuracyTest / N_test
    accuracyTests.append(accuracyTest)

parameters = np.array(Cs).astype(str)
plt.figure(figsize=(10, 10))
# print(rmse_values_training)
plt.plot(parameters, accuracyTrains, marker=".", markersize=10, linestyle="-", color="b", label='training')
plt.plot(parameters, accuracyTests, marker=".", markersize=10, linestyle="-", color="r", label='test')
plt.xlabel("Regularization parameter (C)")
plt.ylabel("Accuracy")
plt.legend(['traininging', 'test'])
plt.show()
