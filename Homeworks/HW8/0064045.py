import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial as spa
from scipy.stats import multivariate_normal

X = np.genfromtxt("hw08_data_set.csv", delimiter=",")
initial_centroids = np.genfromtxt("hw08_initial_centroids.csv", delimiter=",")

N = np.shape(X)[0]
K = np.shape(initial_centroids)[0]
D = np.shape(X)[1]


def update_memberships(centroids, data):
    d = spa.distance_matrix(centroids, data)
    memberships = np.argmin(d, axis=0)
    return memberships


def get_priors(data, memberships):
    return [data[memberships == c].shape[0] / N for c in range(K)]


def get_covariance_matrices(data, memberships, centroids):
    return [get_covariance_matrix(data[memberships == k], centroids[k]) for k in range(K)]


def get_covariance_matrix(new_data, centroid):
    sum_matrix = np.zeros((2, 2))
    for i in range(new_data.shape[0]):
        covariance = np.matmul((new_data[i] - centroid).reshape(2, 1),
                               (new_data[i] - centroid).reshape(1, 2))
        sum_matrix += covariance
    return sum_matrix / new_data.shape[0]


def update_centroids(data, memberships):
    return np.vstack([np.matmul(memberships[k], data) / np.sum(memberships[k], axis=0) for k in range(K)])


def update_covariance(data, membership, mean):
    sum_matrix = np.zeros((2, 2))
    for i in range(N):
        covariance = np.matmul((data[i] - mean).reshape(2, 1),
                               (data[i] - mean).reshape(1, 2)) * membership[i]
        sum_matrix += covariance
    return sum_matrix / np.sum(membership, axis=0)


def update_covariances(data, memberships, means):
    return [update_covariance(data, memberships[k], means[k]) for k in range(K)]


def update_priors(memberships):
    return np.vstack([np.sum(memberships[k], axis=0) / N for k in range(K)])


def create_posterior_probabilities(centroids, covariances, priors):
    return [multivariate_normal(centroids[k], covariances[k]).pdf(X) * priors[k] for k in range(K)]


def plot_current_state(centroids, memberships, data):
    cluster_colors = np.array(["#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a", "#b15928",
                               "#a6cee3", "#b2df8a", "#fb9a99", "#fdbf6f", "#cab2d6", "#ffff99"])
    if memberships is None:
        plt.plot(data[:, 0], data[:, 1], ".", markersize=10, color="black")
    else:
        for c in range(K):
            plt.plot(data[memberships == c, 0], data[memberships == c, 1], ".", markersize=10,
                     color=cluster_colors[c])
    for c in range(K):
        plt.plot(centroids[c, 0], centroids[c, 1], "s", markersize=12,
                 markerfacecolor=cluster_colors[c], markeredgecolor="black")
    plt.xlabel("x1")
    plt.ylabel("x2")


# Initializations
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

centroids = initial_centroids
memberships = update_memberships(centroids, X)
covariances = get_covariance_matrices(X, memberships, centroids)
priors = get_priors(X, memberships)

for i in range(100):
    # Repeated updating of posterior probabilities
    posterior_probabilities = create_posterior_probabilities(centroids, covariances, priors)
    # Repeated updating of memberships
    new_memberships = np.vstack(
        [posterior_probabilities[k] / np.sum(posterior_probabilities, axis=0) for k in range(K)])
    centroids = update_centroids(X, new_memberships)
    covariances = update_covariances(X, new_memberships, centroids)
    priors = update_priors(new_memberships)

# The last posterior probabilities
posterior_probabilities = create_posterior_probabilities(centroids, covariances, priors)
# The last memberships
new_memberships = np.vstack([posterior_probabilities[k] / np.sum(posterior_probabilities, axis=0) for k in range(K)])

print('print(means)')
print(centroids)

memberships = np.argmax(new_memberships, axis=0)

x1_interval = np.linspace(-6, +6, 1201)
x2_interval = np.linspace(-6, +6, 1201)
x, y = np.meshgrid(x1_interval, x2_interval)
pos = np.dstack((x, y))


plt.figure(figsize=(10, 10))
colors = np.array(["#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a", "#b15928",
                               "#a6cee3", "#b2df8a", "#fb9a99"])

for c in range(K):
    predicted_classes = multivariate_normal(centroids[c], covariances[c] * 3).pdf(pos)
    original_classes = multivariate_normal(initial_means[c], initial_covariances[c] * 2).pdf(pos)
    plt.contour(x, y, predicted_classes, levels=1, colors=colors[c])
    plt.contour(x, y, original_classes, levels=1, linestyles="dashed", colors="k")
    plt.plot(X[memberships == c, 0], X[memberships == c, 1], ".", markersize=10,
             color=colors[c])

plt.xlabel("x1")
plt.ylabel("x2")
plt.show()
