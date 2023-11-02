# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 16:50:37 2020

@author: user
"""

import cv2
import numpy as np
import os
import re
import pickle
import time
from sklearn import svm
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random

from skimage import feature

if not os.path.exists('./pickle'):
    os.makedirs('./pickle')

random.seed(0)
def atoi(text):
    return int(text) if text.isdigit() else text



def plot_classifier(kernel, data_train, labels_train, data_test, labels_test, title, file):
    clf = svm.SVC(kernel=kernel)

    start_time = time.time()
    clf.fit(data_train, labels_train)
    elapsed_time = time.time() - start_time
    print("{} seconds".format(elapsed_time))

    pickle.dump(clf, open("./pickle/" + file + ".p", "wb"))

    predict = clf.predict(data_test)

    fpr, tpr, thresholds = roc_curve(labels_test, predict, pos_label=1)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    patch = mpatches.Patch(color='red', label='ROC curve. area = {}, error = {}'.format(np.round(roc_auc, 4),
                                                                                        np.round(1 - roc_auc, 4)))
    plt.legend(handles=[patch], loc='lower right')
    plt.plot(fpr, tpr, color='red')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.savefig(file, dpi=700)
    # plt.show()
    plt.clf()
    
    
    
    

# def recognition(img, classifier, type_norm, step=4):
#     positive_regions = []
#     pyramid = [img]
#     new_level = img

#     while np.shape(new_level)[0] >= 128 and np.shape(new_level)[1] >= 64:
#         # ksize = (size = 3 * 2 * sigma = 1 + 1, size = 3 * 2 * sigma = 1 + 1)
#         new_level = cv2.GaussianBlur(src=new_level, ksize=(7, 7), sigmaX=1)
#         # 0.8333333 is 1 / 1.2
#         new_level = cv2.resize(new_level, dsize=(0, 0), fx=0.8333333, fy=0.8333333)
#         pyramid.append(new_level)

#     for level, img_pyramid in zip(range(len(pyramid)), pyramid):
#         for i in range(0, np.shape(img_pyramid)[0] - 128, step):
#             for j in range(0, np.shape(img_pyramid)[1] - 64, step * 2):
#                 sub_img = cv2.copyMakeBorder(img_pyramid[i:i + 128, j:j + 64], 2, 2, 1, 1, cv2.BORDER_REFLECT)
#                 h = hog(sub_img, type_norm=type_norm)
#                 prediction = classifier.predict(h.reshape(-1, h.shape[0]))
#                 if prediction[0] != 0.0:
#                     positive_regions.append([level, j, i])

#     return positive_regions



    ###########################################################################
    ############################### HOG TRAINING DATA GENERATION #####################################
    ###########################################################################

##############positive#########################

pos_images_train = []
for dirName, subdirList, fileList in os.walk("./INRIAPerson/Train/pos"):
    for fname in sorted(fileList, key=lambda x: [atoi(c) for c in re.split('(\d+)', x)]):
            pos_images_train.append(cv2.imread("./INRIAPerson/Train/pos/" + fname,cv2.IMREAD_GRAYSCALE))
            print(fname)


pos_images_train = [cv2.resize(i, (64,128)) for i in pos_images_train]
start_time = time.time()
hog_positives_train = np.array([ feature.hog(img, orientations=9, pixels_per_cell=(8, 8),
        cells_per_block=(2, 2), transform_sqrt=True, block_norm='L2') for img in pos_images_train])
  
pickle.dump(hog_positives_train, open("./pickle/hog_positives_train_L2.p", "wb"))
elapsed_time = time.time() - start_time
print("TRAIN POSITIVE: {} seconds".format(elapsed_time))

##############negative#########################


neg_images_train = []
for dirName, subdirList, fileList in os.walk("./INRIAPerson/Train/neg"):
    for fname in sorted(fileList, key=lambda x: [atoi(c) for c in re.split('(\d+)', x)]):
        neg_images_train.append(cv2.imread("./INRIAPerson/Train/neg/" + fname,cv2.IMREAD_GRAYSCALE))


neg_images_train = [cv2.resize(i, (64,128)) for i in neg_images_train]
start_time = time.time()
hog_negatives_train = np.array([ feature.hog(img, orientations=9, pixels_per_cell=(8, 8),
        cells_per_block=(2, 2), transform_sqrt=True, block_norm='L2') for img in neg_images_train])
  
pickle.dump(hog_negatives_train, open("./pickle/hog_negatives_train_L2.p", "wb"))
elapsed_time = time.time() - start_time
print("TRAIN NEGATIVE: {} seconds".format(elapsed_time))


    ###########################################################################
    ############################### HOG TEST DATA GENERATION #####################################
    ###########################################################################

##############positive#########################
pos_images_test = []
for dirName, subdirList, fileList in os.walk("./INRIAPerson/Test/pos"):
    for fname in sorted(fileList, key=lambda x: [atoi(c) for c in re.split('(\d+)', x)]):
            pos_images_test.append(cv2.imread("./INRIAPerson/Test/pos/" + fname,cv2.IMREAD_GRAYSCALE))


pos_images_test = [cv2.resize(i, (64,128)) for i in pos_images_test]
start_time = time.time()
hog_positives_test = np.array([ feature.hog(img, orientations=9, pixels_per_cell=(8, 8),
        cells_per_block=(2, 2), transform_sqrt=True, block_norm='L2') for img in pos_images_test])
  
pickle.dump(hog_positives_train, open("./pickle/hog_positives_test_L2.p", "wb"))
elapsed_time = time.time() - start_time
print("TEST POSITIVE: {} seconds".format(elapsed_time))

##############negative#########################

neg_images_test = []
for dirName, subdirList, fileList in os.walk("./INRIAPerson/Test/neg"):
    for fname in sorted(fileList, key=lambda x: [atoi(c) for c in re.split('(\d+)', x)]):
            neg_images_test.append(cv2.imread("./INRIAPerson/Test/neg/" + fname,cv2.IMREAD_GRAYSCALE))


neg_images_test = [cv2.resize(i, (64,128)) for i in neg_images_test]
start_time = time.time()
hog_negatives_test = np.array([ feature.hog(img, orientations=9, pixels_per_cell=(8, 8),
        cells_per_block=(2, 2), transform_sqrt=True, block_norm='L2') for img in neg_images_test])
  
pickle.dump(hog_negatives_test, open("./pickle/hog_negatives_test_L2.p", "wb"))
elapsed_time = time.time() - start_time
print("TEST NEGATIVE: {} seconds".format(elapsed_time))

#############################################################    
######################classifier#############################
#############################################################

####################load data################################
hog_positives_train_L2 = pickle.load(open("./pickle/hog_positives_train_L2.p", "rb"))
hog_negatives_train_L2 = pickle.load(open("./pickle/hog_negatives_train_L2.p", "rb"))

hog_positives_test_L2 = pickle.load(open("./pickle/hog_positives_test_L2.p", "rb"))
hog_negatives_test_L2 = pickle.load(open("./pickle/hog_negatives_test_L2.p", "rb"))

pos_labels_train = np.ones(len(hog_positives_train_L2))
neg_labels_train = np.zeros(len(hog_negatives_train_L2))
labels_train = np.append(pos_labels_train, neg_labels_train)
   
pos_labels_test = np.ones(len(hog_positives_test_L2))
neg_labels_test = np.zeros(len(hog_negatives_test_L2))
labels_test = np.append(pos_labels_test, neg_labels_test)

data_train_L2 = np.append(hog_positives_train_L2, hog_negatives_train_L2, axis=0)
data_test_L2 = np.append(hog_positives_test_L2, hog_negatives_test_L2, axis=0)


plot_classifier('linear', data_train_L2, labels_train,data_test_L2, labels_test,'Norm: L2.', "trained_linear_svm_L2")
