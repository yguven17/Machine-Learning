import math
import warnings
import numpy as np
import pandas as pd
import csv
from matplotlib import pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)


# to avoid log functions errors
def logCalculator(x):
    return np.log(x + 1e-100)


# rad data from cs files
images_set = np.genfromtxt("hw02_data_set_images.csv", delimiter=",")
labels_set = np.genfromtxt("hw02_data_set_labels.csv", delimiter=",")

images_training_set = []
label_training_set = []
images_test_set = []
label_test_set = []

# separate test and training data
for i in range(5):

    for j in range(25):
        images_training_set.append(images_set[j + 39 * i])
        label_training_set.append(labels_set[j + 39 * i])

    for k in range(14):
        images_test_set.append(images_set[25 + k + 39 * i])
        label_test_set.append(labels_set[25 + k + 39 * i])

images_training_set = np.array(images_training_set)
label_training_set = np.array(label_training_set)
images_test_set = np.array(images_test_set)
label_test_set = np.array(label_test_set)

# print(images_training_set)
# print(label_training_set)
# print(images_test_set)
# print(label_test_set)


# find k and n from sets
K = (len(np.unique(label_test_set)))
N = images_training_set.shape[0]

# calculate and print pcd
print("\nprint(pcd)")
sample_means = np.stack(np.nanmean(images_training_set[label_training_set == (i + 1)], axis=0) for i in range(K))
print(sample_means)

# calculate and print class priors
print("\nprint(class_priors)")
class_priors = [np.nanmean(label_training_set == (i + 1)) for i in range(K)]
print(class_priors)


# calculate score for datas
def scoreCalculator(x, sample_mean, class_prior):
    score_values = np.array([0, 0, 0, 0, 0])
    for i in range(K):
        score_values[i] = np.sum(np.dot(np.transpose(x), logCalculator(sample_mean[i])) + np.dot((1 - np.transpose(x)),
                                                                                                 logCalculator(
                                                                                                     1 - sample_mean[
                                                                                                         i]))) + np.log(
            class_prior[i])
    return score_values


# g max calculator
def gmaxFinder(predicted_values, g_scores):
    for i in range(len(g_scores)):
        gmax = np.max(g_scores[i])
        if g_scores[i][0] == gmax:
            predicted_values.append(1)
        elif g_scores[i][1] == gmax:
            predicted_values.append(2)
        elif g_scores[i][2] == gmax:
            predicted_values.append(3)
        elif g_scores[i][3] == gmax:
            predicted_values.append(4)
        else:
            predicted_values.append(5)
    predicted_values = np.array(predicted_values)
    return predicted_values


# calculate g for training data
trainnig_g_score = [scoreCalculator(images_training_set[i], sample_means, class_priors) for i in
                    range(np.shape(images_training_set)[0])]
# calculate g for test data
test_g_score = [scoreCalculator(images_test_set[i], sample_means, class_priors) for i in
                range(np.shape(images_test_set)[0])]

# make predictions from g scores using max finder
trainning_prediction = gmaxFinder([], trainnig_g_score)
test_prediction = gmaxFinder([], test_g_score)

# calculate and print confusion matrix
print("\nprint(confusion_matrix)")
confusion_matrix = pd.crosstab(trainning_prediction, label_training_set, rownames=["y_pred"], colnames=["y_truth"])
print(confusion_matrix)

# calculate and print confusion matrix
print("\nprint(confusion_matrix)")
confusion_matrix = pd.crosstab(test_prediction, label_test_set, rownames=["y_pred"], colnames=["y_truth"])
print(confusion_matrix)
