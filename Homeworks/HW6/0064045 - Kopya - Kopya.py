import cvxopt as cvx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.spatial.distance as dt
from sklearn.metrics import accuracy_score

# In[260]:


# read data into memory
X = np.genfromtxt("hw06_data_set_images.csv", delimiter=",")
y = np.genfromtxt("hw06_data_set_labels.csv").astype(int)

# In[261]:


X_train = X[:1000, :]
X_test = X[1000:, :]
y_train = y[:1000]
y_test = y[1000:]

# get number of samples and number of features
N_train = len(y_train)
N_test = len(y_test)
D = X.shape[1]


# In[ ]:


# In[262]:


# define Gaussian kernel function
def gaussian_kernel(X1, X2, s):
    D = dt.cdist(X1, X2)
    K = np.exp(-D ** 2 / (2 * s ** 2))
    return (K)


# In[263]:


# set learning parameters
C = 10
epsilon = 1e-3
s = 10


# In[264]:


def kernal_machine(K, y, C, N, D, epsilon):
    # calculate Gaussian kernel
    yyK = np.matmul(y[:, None], y[None, :]) * K

    P = cvx.matrix(yyK)
    q = cvx.matrix(-np.ones((N, 1)))
    G = cvx.matrix(np.vstack((-np.eye(N), np.eye(N))))
    h = cvx.matrix(np.vstack((np.zeros((N, 1)), C * np.ones((N, 1)))))
    A = cvx.matrix(1.0 * y[None, :])
    b = cvx.matrix(0.0)

    # use cvxopt library to solve QP problems
    result = cvx.solvers.qp(P, q, G, h, A, b)
    alpha = np.reshape(result["x"], N)
    alpha[alpha < C * epsilon] = 0
    alpha[alpha > C * (1 - epsilon)] = C

    # find bias parameter
    support_indices, = np.where(alpha != 0)
    active_indices, = np.where(np.logical_and(alpha != 0, alpha < C))
    w0 = np.mean(
        y[active_indices] * (1 - np.matmul(yyK[np.ix_(active_indices, support_indices)], alpha[support_indices])))

    return w0, alpha


# In[265]:


K_test = gaussian_kernel(X_test, X_train, s)
gscore_test = np.array([])
K_train = gaussian_kernel(X_train, X_train, s)
gscore_train = np.array([])
for i in range(np.max(y)):
    y_train_temp = y_train.copy()
    y_train_temp[y_train_temp[:] != i + 1] = -1
    y_train_temp[y_train_temp[:] == i + 1] = 1
    w0, alpha = kernal_machine(K_train, y_train_temp, C, N_train, D, epsilon)
    # calculate predictions on training samples
    f_predicted_train = np.matmul(K_train, y_train_temp[:, None] * alpha[:, None]) + w0
    f_predicted_test = np.matmul(K_test, y_train_temp[:, None] * alpha[:, None]) + w0
    gscore_train = np.append(gscore_train, f_predicted_train)
    gscore_test = np.append(gscore_test, f_predicted_test)

# In[266]:


y_train_predicted = np.argmax(gscore_train.reshape(9, 1000), axis=0) + 1
y_test_predicted = np.argmax(gscore_test.reshape(9, 2000), axis=0) + 1

# In[267]:


confusion_matrix = pd.crosstab(np.reshape(y_train_predicted, N_train), y_train, rownames=['y_predicted'],
                               colnames=['y_train'])
print(confusion_matrix)

# In[268]:


confusion_matrix = pd.crosstab(np.reshape(y_test_predicted, N_test), y_test, rownames=['y_predicted'],
                               colnames=['y_test'])
print(confusion_matrix)

# In[269]:


regularization_parameters = np.array([pow(10, -1), pow(10, (-0,5)), pow(10, 0), pow(10, (0,5)), pow(10, 1), pow(10, (1,5)), pow(10, 2),
            pow(10, (2,5)), pow(10, 3)])

# In[273]:


accuracy_train_storer = np.array([])
accuracy_test_storer = np.array([])

for i in regularization_parameters:
    gscore_train = np.array([])
    gscore_test = np.array([])
    for j in range(np.max(y)):
        y_train_temp = y_train.copy()
        y_train_temp[y_train_temp[:] != j + 1] = -1
        y_train_temp[y_train_temp[:] == j + 1] = 1
        w0, alpha = kernal_machine(K_train, y_train_temp, i, N_train, D, epsilon)
        f_predicted_train = np.matmul(K_train, y_train_temp[:, None] * alpha[:, None]) + w0
        f_predicted_test = np.matmul(K_test, y_train_temp[:, None] * alpha[:, None]) + w0
        gscore_train = np.append(gscore_train, f_predicted_train)
        gscore_test = np.append(gscore_test, f_predicted_test)

    y_train_predicted = np.argmax(gscore_train.reshape(5, 1000), axis=0) + 1
    y_test_predicted = np.argmax(gscore_test.reshape(5, 2000), axis=0) + 1
    accuracy_train_storer = np.append(accuracy_train_storer, accuracy_score(y_train, y_train_predicted))
    accuracy_test_storer = np.append(accuracy_test_storer, accuracy_score(y_test, y_test_predicted))

# In[299]:


plt.figure(figsize=(10, 10))
plt.plot(regularization_parameters, accuracy_train_storer, "bo-")
plt.plot(regularization_parameters, accuracy_test_storer, "ro-")
plt.xscale("log")
plt.xlabel("Regularization parameters (C)")
plt.ylabel("Accuracy")
plt.legend(["training", "test"], loc="upper right")