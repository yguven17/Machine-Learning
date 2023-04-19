import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

images_data = np.genfromtxt("hw07_data_set_images.csv", delimiter=",")
labels_data = np.genfromtxt("hw07_data_set_labels.csv", delimiter=",")

x_training = images_data[:2000, :]
x_test = images_data[2000:, :]
y_training = labels_data[:2000]
y_test = labels_data[2000:]

classes = np.unique(y_training)
K = len(classes)
N = x_training.shape[0]
D = x_training.shape[1]

sample_mean = np.mean(x_training, axis=0)
SW = np.zeros((784, 784))
SB = np.zeros((784, 784))

for c in classes:
    Xc = x_training[y_training == c]
    meanc = np.mean(Xc, axis=0)

    SW += (Xc - meanc).T.dot((Xc - meanc))
    n_c = Xc.shape[0]

    SB += n_c * ((meanc - sample_mean).reshape(784, 1)).dot((meanc - sample_mean).reshape(784, 1).T)

print("print(SW[0:4, 0:4])")
print(SW[0:4, 0:4])
print()
print("print(SB[0:4, 0:4])")
print(SB[0:4, 0:4])
print()

SWSB = np.linalg.inv(SW).dot(SB)

eigenval, eigenvec = np.linalg.eig(SWSB)
values = np.real(eigenval)
eigenvec = eigenvec.T
vectors = np.real(eigenvec)

print("print(values[0:9])")
print(values[0:9])
print()

x_training_projected = np.dot(x_training, vectors[0: 2].T)

plt.size = (10, 10)
point_colors = np.array(
    ["#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a", "#a6cee3", "#b2df8a", "#fb9a99", "#fdbf6f", "#cab2d6"])
for c in range(K):
    plt.plot(x_training_projected[y_training == c + 1, 0], x_training_projected[y_training == c + 1, 1], marker="o", markersize=4,
             linestyle="none", color=point_colors[c])
plt.legend(["t-shirt/top", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankle boot"],
           loc="upper left", markerscale=2)
plt.xlim(-6, 6)
plt.ylim(-6, 6)
plt.xlabel("Component#1")
plt.ylabel("Component#2")
plt.show()

x_test_projected = np.dot(x_test, vectors[0: 2].T)

plt.size = (10, 10)
for c in range(K):
    plt.plot(x_test_projected[y_test == c + 1, 0], x_test_projected[y_test == c + 1, 1], marker="o", markersize=4,
             linestyle="none", color=point_colors[c])
plt.legend(["t-shirt/top", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankle boot"],
           loc="upper left", markerscale=2)
plt.xlim(-6, 6)
plt.ylim(-6, 6)
plt.xlabel("Component#1")
plt.ylabel("Component#2")
plt.show()

x_training_projected_9d = np.dot(x_training, vectors[0: 9].T)
x_test_projected_9d = np.dot(x_test, vectors[0: 9].T)

𝑘_nearest_neighbor = KNeighborsClassifier(n_neighbors=11)
𝑘_nearest_neighbor.fit(x_training_projected_9d, y_training)

x_training_projected_9d = 𝑘_nearest_neighbor.predict(x_training_projected_9d)
x_test_projected_9d = 𝑘_nearest_neighbor.predict(x_test_projected_9d)

confusion_matrix_train = pd.crosstab(np.reshape(x_training_projected_9d, N), y_training, rownames=["y_hat"], colnames=["y_train"])
confusion_matrix_test = pd.crosstab(np.reshape(x_test_projected_9d, N), y_test, rownames=["y_hat"], colnames=["y_test"])

print("print(confusion_matrix)")
print(confusion_matrix_train)
print()
print("print(confusion_matrix)")
print(confusion_matrix_test)
print()
