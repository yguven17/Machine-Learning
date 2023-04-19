import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
eta = 1e-3
epsilon = 1e-3


train=[]
test=[]
train_truth=[]
test_truth=[]
data_set = np.genfromtxt("hw03_data_set_images.csv", delimiter = ",")
labels = (np.genfromtxt("hw03_data_set_labels.csv", delimiter=",", dtype=str)).astype(str)
K = 5
N = data_set.shape[0]
classes = np.unique(labels)




for i in range(N):
    result = np.argwhere(classes==labels[i])
    labels[i]=result[0][0]
labels= labels.astype(np.int)



for i in range(0,5):
    train[(39*i):(25+39*i)]=data_set[(39*i):(25+39*i)]
    test[(25+i*39):39*(i+1)]=data_set[(25+i*39):39*(i+1)]
    train_truth[(39*i):(25+39*i)]= labels[(39*i):(25+39*i)]
    test_truth[(25+i*39):39*(i+1)] = labels[(25+i*39):39*(i+1)]
train=np.array(train)
test=np.array(test)
train_truth=np.array(train_truth)
test_truth=np.array(test_truth)





# one-hot encoding
Train_truth = np.zeros((len(train), K)).astype(int)

for i in range(len(train)):
    Train_truth[i][train_truth[i]]=1
print(Train_truth)

def gradient_W(X, Y_truth, Y_predicted):
    return (np.asarray([np.matmul((Y_truth[:,c] - Y_predicted[:,c])*(Y_predicted[:,c])*(Y_predicted[:,c] - 1),X) for c in range(K)]).transpose())


# define sigmoid function
def sigmoid(X, w, w0):
    return(1 / (1 + np.exp(-(np.matmul( w.T, X.T) + w0.T))))


def gradient_w0(Y_truth, Y_predicted):
    return np.sum((Y_truth - Y_predicted)*Y_predicted*(Y_predicted - 1), axis=0)



# randomly initalize
np.random.seed(521)
w = np.random.uniform(low = -0.01, high = 0.01, size = (train.shape[1],K))
w0 = np.random.uniform(low = -0.01, high = 0.01, size = (1,K))


# iterative
iteration = 1
objective_values = []
while 1:
    train_predicted = sigmoid(train, w, w0)
    objective_values = np.append(objective_values,0.5*np.sum((Train_truth-train_predicted.T)**2))

    w0_old = w0
    w_old = w

    w = w - eta * gradient_W(train, Train_truth, train_predicted.T)
    w0 = w0 - eta * gradient_w0(Train_truth, train_predicted.T)

    if np.sqrt(np.sum((w0 - w0_old)**2) + np.sum((w - w_old)**2)) < epsilon:
        break

    iteration = iteration + 1

print(w, w0)


plt.figure(figsize = (10, 6))
plt.plot(range(1, iteration + 1), objective_values, "k-")
plt.xlabel("Iteration")
plt.ylabel("Error")
plt.show()


# calculate confusion matrix
train_predicted = np.argmax(train_predicted, axis = 0) + 1
confusion_matrix = pd.crosstab(train_predicted, train_truth+1, rownames = ['train_pred'], colnames = ['train_truth'])
print(confusion_matrix)

# calculate confusion matrix
test_predicted = sigmoid(test,w,w0)
test_predicted = np.argmax(test_predicted, axis = 0) + 1
confusion_matrix = pd.crosstab(test_predicted, test_truth+1, rownames = ['test_pred'], colnames = ['test_truth'])
print(confusion_matrix)