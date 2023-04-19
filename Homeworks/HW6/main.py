import cvxopt as cvx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.spatial.distance as dt



images_data = np.genfromtxt("hw06_data_set_images.csv", delimiter = ",")
labels_data = np.genfromtxt("hw06_data_set_labels.csv", delimiter = ",")


x_train = images_data[:1000,:]
x_test = images_data[1000:,:]
y_train = (labels_data[:1000]).astype(int)
y_test = (labels_data[1000:]).astype(int)
N_train = len(y_train)
N_test = N_train

H_train=[]
for i in range (1000):
    hist, bin_edges = np.histogram((x_train[i]), bins=64);
    H_train.append((hist/784))
H_train = np.array(H_train)
print("H_train and H_test")
print(H_train[0:5,0:5])


H_test=[]
for i in range (1000):
    hist, bin_edges = np.histogram((x_test[i]), bins=64);
    H_test.append((hist/784))
H_test = np.array(H_test)
print(H_test[0:5,0:5])


def kernel(h1, h2):
    result=[]
    for j in range(1000):
        sums = []
        for k in range(1000):
            sm = 0
            for i in range(64):
                sm += (min(h1[j][i], h2[k][i]))
            sums.append(sm)
        result.append(sums)
    result=np.array(result)
    return(result)

K_train = kernel(H_train, H_train)
K_test = kernel(H_test,H_train)
print("K_train and K_test")
print(K_train[0:5])
print(K_test[0:5])


def classifier(y_train, c, N_train):
    yyK = np.matmul(y_train[:,None], y_train[None,:]) * K_train
    
    # set learning parameters
    C = c
    epsilon = 0.001
    
    P = cvx.matrix(yyK)
    q = cvx.matrix(-np.ones((N_train, 1)))
    G = cvx.matrix(np.vstack((-np.eye(N_train), np.eye(N_train))))
    h = cvx.matrix(np.vstack((np.zeros((N_train, 1)), C * np.ones((N_train, 1)))))
    A = cvx.matrix(1.0 * y_train[None,:])
    b = cvx.matrix(0.0)
                        
    # use cvxopt library to solve QP problems
    result = cvx.solvers.qp(P, q, G, h, A, b, options={'show_progress': False})
    alpha = np.reshape(result["x"], N_train)
    alpha[alpha < C * epsilon] = 0
    alpha[alpha > C * (1 - epsilon)] = C
    
    # find bias parameter
    support_indices, = np.where(alpha != 0)
    active_indices, = np.where(np.logical_and(alpha != 0, alpha < C))
    w0 = np.mean(y_train[active_indices] * (1 - np.matmul(yyK[np.ix_(active_indices, support_indices)], alpha[support_indices])))
    return alpha, w0


alpha, w0 = classifier(y_train,10,N_train)
alpha2,w02 = classifier(y_test,10,N_test)
def predictions(y_train, K_train, w0, alpha):
    # calculate predictions on training samples
    f_predicted = np.matmul(K_train, y_train[:,None] * alpha[:,None]) + w0
    
    # calculate confusion matrix
    y_predicted = 2 * (f_predicted > 0.0) - 1
    confusion_matrix = pd.crosstab(np.reshape(y_predicted, N_train), y_train,
                                   rownames = ["y_predicted"], colnames = ["y_train"])
    print(confusion_matrix)

predictions(y_train, K_train, w0, alpha)


f_predicted2 = np.matmul(K_test, y_train[:,None] * alpha[:,None]) + w0

# calculate confusion matrix
y_predicted2 = 2 * (f_predicted2 > 0.0) - 1
confusion_matrix = pd.crosstab(np.reshape(y_predicted2, N_train), y_test,
                               rownames = ["y_predicted"], colnames = ["y_train"])
print(confusion_matrix)


list_C =[10**(-1), 10**(-0.5), 10**(0), 10**(0.5),10**(1) ,10**(1.5), 10**(2), 10**(2.5), 10**(3)]
train_accuracy = []
test_accuracy = []
label_train = y_train
label_test = y_test
for c in list_C:
    
    alpha, w0 = classifier(y_train,c,N_train)
    f_predicted_train = np.matmul(K_train, y_train[:,None] * alpha[:,None]) + w0
    label_predicted_train = 2 * (f_predicted_train > 0.0) - 1
    

    f_predicted_test = np.matmul(K_test, y_train[:,None] * alpha[:,None]) + w0
    label_predicted_test = 2 * (f_predicted_test > 0.0) - 1
 
    counter = 0
    for i in range(N_train):
        if label_predicted_train[i] == label_train[i]:
            counter +=1
    
    accuracy1 = (counter / N_train)
    train_accuracy.append(accuracy1) 
    
    counter = 0
    for i in range(N_test):
        if label_predicted_test[i] == label_test[i]:
            counter +=1
            
    accuracy2 = (counter / N_test)
    test_accuracy.append(accuracy2) 
    
    
plt.figure(figsize = (10, 10))
plt.plot(list_C, train_accuracy, marker = ".", markersize = 10, linestyle = "-", color = "b",label='train')
plt.plot(list_C, test_accuracy, marker = ".", markersize = 10, linestyle = "-", color = "r",label='test')
plt.xscale('log')
plt.legend(['training', 'test'])
plt.ylabel("Accuracy ")
plt.xlabel("Regularization parameter (C)")
plt.show()
    


