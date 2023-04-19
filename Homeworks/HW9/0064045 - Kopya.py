import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial as spa
from scipy.linalg import sqrtm

X = np.genfromtxt("hw09_data_set.csv", delimiter=",")

initial_means = np.array([[+5.0, +5.0],
                          [-5.0, +5.0],
                          [-5.0, -5.0],
                          [+5.0, -5.0],
                          [+5.0, +0.0],
                          [+0.0, +5.0],
                          [-5.0, +0.0],
                          [+0.0, -5.0],
                          [+0.0, +0.0]])

initial_covariances = np.array([[[+0.8, -0.6], [-0.6, +0.8]],
                                [[+0.8, +0.6], [+0.6, +0.8]],
                                [[+0.8, -0.6], [-0.6, +0.8]],
                                [[+0.8, +0.6], [+0.6, +0.8]],
                                [[+0.2, +0.0], [+0.0, +1.2]],
                                [[+1.2, +0.0], [+0.0, +0.2]],
                                [[+0.2, +0.0], [+0.0, +1.2]],
                                [[+1.2, +0.0], [+0.0, +0.2]],
                                [[+1.6, +0.0], [+0.0, +1.6]]])

N = np.shape(X)[0]
K = np.shape(initial_means)[0]
D = np.shape(X)[1]

distance_const = 2.0

x1 = list()
x2 = list()

for pair in X:
    x1.append(pair[0])
    x2.append(pair[1])

plt.figure(figsize=(10, 10))
plt.plot(x1, x2, ".", markersize=10, color="k")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()


def get_euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def update_centroids(memberships, X):
    if memberships is None:
        centroids = np.vstack([Z[242], Z[528], Z[570], Z[590], Z[648], Z[667], Z[774], Z[891], Z[955]])
    else:
        centroids = np.vstack([np.mean(X[memberships == k, :], axis=0) for k in range(K)])
    return centroids


def update_memberships(centroids, X):
    d = spa.distance_matrix(centroids, X)
    memberships = np.argmin(d, axis=0)
    return memberships


B = np.arange(N ** 2).reshape(N, N)
D = np.arange(N ** 2).reshape(N, N)

plt.figure(figsize=(10, 10))

for i in range(N):
    p1 = X[i]
    for j in range(N):
        if i == j:
            B[i][j] = 0
            continue
        p2 = X[j]
        distance = get_euclidean_distance(p1, p2)
        if distance < distance_const:
            B[i][j] = 1
            x_values = [p1[0], p2[0]]
            y_values = [p1[1], p2[1]]
            plt.plot(x_values, y_values, color="gray")
        else:
            B[i][j] = 0

plt.plot(x1, x2, ".", markersize=10, color="k")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()

for i in range(N):
    summing = 0
    for j in range(N):
        summing += B[i][j]
        D[i][j] = 0
    D[i][i] = summing

L = D - B

print("B Matrix:")
print(B)
print("D Matrix:")
print(D)
print("L Matrix:")
print(L)

D_square_root = sqrtm(D)
Dsr_inverse = np.linalg.inv(D_square_root)

identity = np.identity(N)
L_symmetric = identity - np.matmul(Dsr_inverse, np.matmul(B, Dsr_inverse))
R = 5

eigvals, eigvecs = np.linalg.eig(L_symmetric)

sorted_eigvals = np.argsort(eigvals)
# First element is 0 so I ignored it.
smallest_eigvals = sorted_eigvals[1:R + 1]
Z = eigvecs[:, smallest_eigvals]

centroids = update_centroids(None, Z)
memberships = update_memberships(centroids, Z)
iteration = 1
while True:
    old_centroids = centroids
    centroids = update_centroids(memberships, X)
    if np.alltrue(centroids == old_centroids):
        break
    old_memberships = memberships
    memberships = update_memberships(centroids, X)
    iteration = iteration + 1

print("Iteration count: {}".format(iteration))

colors = np.array(["#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a", "#b15928",
                   "#a6cee3", "#b2df8a", "#fb9a99", "#fdbf6f", "#cab2d6", "#ffff99"])
plt.figure(figsize=(10, 10))
for c in range(K):
    plt.plot(X[memberships == c, 0], X[memberships == c, 1], ".", markersize=10, color=colors[c])
    plt.plot(centroids[c, 0], centroids[c, 1], "s", markersize=10, markeredgecolor="k", color=colors[c])

plt.xlabel("x1")
plt.ylabel("x2")
plt.show()
