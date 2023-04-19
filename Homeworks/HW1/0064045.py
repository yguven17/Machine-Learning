import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as linalg
import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

np.random.seed(521)
# Generating random data points from given intervals
#Then prints
mean1 = np.array([0, 4.5])
mean2 = np.array([-4.5, -1])
mean3 = np.array([4.5, -1])
mean4 = np.array([0, -4])
class_means = np.array([mean1, mean2, mean3, mean4])
#print(class_means)

deviation1 = np.array([np.array([3.2, 0]), np.array([0, 1.2])])
deviation2 = np.array([np.array([1.2, 0.8]), np.array([0.8, 1.2])])
deviation3 = np.array([np.array([1.2, -0.8]), np.array([-0.8, 1.2])])
deviation4 = np.array([np.array([1.2, 0]), np.array([0, 3.2])])
class_deviations = np.array([deviation1, deviation2, deviation3, deviation4])
#print(class_deviations)

class_sizes = np.array([105, 145, 135, 115])

points1 = np.random.multivariate_normal(class_means[0], class_deviations[0], class_sizes[0])
points2 = np.random.multivariate_normal(class_means[1], class_deviations[1], class_sizes[1])
points3 = np.random.multivariate_normal(class_means[2], class_deviations[2], class_sizes[2])
points4 = np.random.multivariate_normal(class_means[3], class_deviations[3], class_sizes[3])

X = np.concatenate((points1, points2, points3, points4))
y = np.concatenate((np.repeat(1, class_sizes[0]), np.repeat(2, class_sizes[1]), np.repeat(3, class_sizes[2]),
                    np.repeat(4, class_sizes[3])))
K = len(class_sizes)

# Plot Data Points
plt.figure("Data Points", figsize=(8, 8))
plt.suptitle("Data Points")
plt.xlim(-8, 8)
plt.ylim(-8, 8)

plt.plot(X[y == 1, 0], X[y == 1, 1], "r.", markersize=10)
plt.plot(X[y == 2, 0], X[y == 2, 1], "g.", markersize=10)
plt.plot(X[y == 3, 0], X[y == 3, 1], "b.", markersize=10)
plt.plot(X[y == 4, 0], X[y == 4, 1], "m.", markersize=10)

plt.xlabel("x1")
plt.ylabel("x2")
plt.show()

# Estimations
# Mean Values
print("Sample Means:")
sample_means = np.stack(np.mean(X[y == (c + 1)], axis=0) for c in range(K))
print(sample_means)
print()

# Covariance Values
print("Sample Covariance:")
sum1 = np.zeros(2)
sum2 = np.zeros(2)
sum3 = np.zeros(2)
sum4 = np.zeros(2)
for i in range(class_sizes[0]):
    diff = points1[i] - sample_means[0]
    diff = np.vstack(diff)
    cov = np.matmul(diff, np.transpose(diff))
    sum1 = sum1 + cov

for i in range(class_sizes[1]):
    diff = points2[i] - sample_means[1]
    diff = np.vstack(diff)
    cov = np.matmul(diff, np.transpose(diff))
    sum2 = sum2 + cov

for i in range(class_sizes[2]):
    diff = points3[i] - sample_means[2]
    diff = np.vstack(diff)
    cov = np.matmul(diff, np.transpose(diff))
    sum3 = sum3 + cov

for i in range(class_sizes[3]):
    diff = points4[i] - sample_means[3]
    diff = np.vstack(diff)
    cov = np.matmul(diff, np.transpose(diff))
    sum4 = sum4 + cov

covariance = np.array([sum1 / class_sizes[0], sum2 / class_sizes[1], sum3 / class_sizes[2], sum4 / class_sizes[3]])
print(covariance)
print()

# Class Priors
print("Class Priors:")
class_priors = [np.mean(y == (c + 1)) for c in range(K)]
print(class_priors)
print()

# Classification,
D = 2
W = np.stack(-0.5 * linalg.cho_solve(linalg.cho_factor(covariance[c]), np.eye(2)) for c in range(K))
w = np.vstack(
    np.matmul(linalg.cho_solve(linalg.cho_factor(covariance[c]), np.eye(2)), sample_means[c]) for c in range(K))
w0 = np.stack(-0.5 * np.matmul(
    np.matmul(np.transpose(sample_means[c]), linalg.cho_solve(linalg.cho_factor(covariance[c]), np.eye(2))),
    sample_means[c]) - \
              D / 2 * np.log(math.pi * 2) - 0.5 * np.log(np.linalg.det(covariance[c])) + np.log(class_priors[c]) for c
              in range(K))

y_predicted = []
y_truth = []

for i in range(len(X)):
    g = np.stack(
        np.matmul(np.matmul(np.transpose(X[i]), W[c]), X[i]) + np.matmul(np.transpose(w[c]), X[i]) + w0[c]
        for c in range(K))

    g_max = max(g[0], g[1], g[2], g[3])
    y_truth.append(y[i])
    if g_max == g[0]:
        y_predicted.append(1)
    elif g_max == g[1]:
        y_predicted.append(2)
    elif g_max == g[2]:
        y_predicted.append(3)
    else:
        y_predicted.append(4)

y_predicted = np.array(y_predicted)
y_truth = np.array(y_truth)

confusion_matrix = pd.crosstab(y_predicted, y, rownames=['y_pred'], colnames=['y_truth'])
print(confusion_matrix)

# Visualization

frequency = 401
x1_interval = np.linspace(-6, +6, frequency)
x2_interval = np.linspace(-6, +6, frequency)
x1_grid, x2_grid = np.meshgrid(x1_interval, x2_interval)


def predict(temp):
    g = np.stack(
        np.matmul(np.matmul(np.transpose(temp[0]), W[c]), temp[0]) + np.matmul(np.transpose(w[c]), temp[0]) + w0[c]
        for c in range(K))
    g_max = max(g[0], g[1], g[2], g[3])

    if g_max == g[0]:
        return 1
    elif g_max == g[1]:
        return 2
    elif g_max == g[2]:
        return 3
    else:
        return 4


plt.figure("Decision boundaries", figsize=(8, 8))
plt.suptitle("Decision Boundaries")
plt.xlim(-8, 8)
plt.ylim(-8, 8)
plt.plot(X[y_truth == 1, 0], X[y_truth == 1, 1], "r.", markersize=10)
plt.plot(X[y_truth == 2, 0], X[y_truth == 2, 1], "g.", markersize=10)
plt.plot(X[y_truth == 3, 0], X[y_truth == 3, 1], "b.", markersize=10)
plt.plot(X[y_truth == 4, 0], X[y_truth == 4, 1], "m.", markersize=10)
plt.plot(X[y_predicted != y_truth, 0], X[y_predicted != y_truth, 1], "ko", markersize=12, fillstyle="none")

z = np.array([predict(np.array([[x1, x2]])) for x1, x2 in zip(np.ravel(x1_grid), np.ravel(x2_grid))])
Z = z.reshape(x1_grid.shape)

contourplot = plt.contour(x1_grid, x2_grid, Z, 0, colors=['r', 'b', 'g', 'm'])
plt.imshow(Z, origin='lower', extent=[-8, 8, -8, 8], alpha=0.1)

plt.xlabel("x1")
plt.ylabel("x2")
plt.show()