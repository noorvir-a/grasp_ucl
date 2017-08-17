"""
author: Noorvir Aulakh
date: 17 July 2017

N.B. Parts of this file are adopted from (Mahler, 2017)

Mahler, Jeffrey, Jacky Liang, Sherdil Niyaz, Michael Laskey, Richard Doan, Xinyu Liu, Juan Aparicio Ojea, and
Ken Goldberg. "Dex-Net 2.0: Deep Learning to Plan Robust Grasps with Synthetic Point Clouds and Analytic Grasp Metrics."
arXiv preprint arXiv:1703.09312 (2017).

"""

from gqcnn import GQCNN, SGDOptimizer
from autolab_core import YamlConfig
from grasp_ucl.neural_networks.guant import GUANt
from grasp_ucl.utils.pre_process import DataLoader
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas
import logging
import time



# get architecture to use
# load model
# load raw depth image and sample grasps using policy.py
#


####################
# GUANt Test
####################

guant_config = YamlConfig('/home/noorvir/catkin_ws/src/grasp_ucl/cfg/guant.yaml')
dataset_config = YamlConfig(guant_config['dataset_config'])['guant']

guant = GUANt(guant_config)

# store metrics over multiple trails in lists
accuracy_list = []
error_list = []
gt_labels_list = []
predicted_labels_list = []

test_data_loader = DataLoader(guant, dataset_config)
num_test_trials = 2

for trial in xrange(num_test_trials):

    img_batch, pose_batch, label_batch = test_data_loader.get_test_batch()

    print('Starting trial number %d of %d' % (trial, num_test_trials))
    if trial == num_test_trials -1:
        accuracy, error, predicted_labels = guant.predict(img_batch, pose_batch, label_batch, close_sess=True, is_test=True)
    else:
        accuracy, error, predicted_labels = guant.predict(img_batch, pose_batch, label_batch, is_test=True)

    gt_labels = np.where(label_batch[:, :] == 1)[1]

    accuracy_list.append(accuracy)
    error_list.append(error)
    gt_labels_list.append(list(gt_labels))
    predicted_labels_list.append(list(predicted_labels))

print('Finished all trials')

label_batch_pd = pandas.Series(gt_labels_list, name='Actual')
predicted_labels_pd = pandas.Series(predicted_labels_list, name='Predicted')

# confusion_mat = pandas.crosstab(label_batch_pd, predicted_labels_pd, rownames=['Actual'], colnames=['Predicted'],  margins=True)
confusion_mat = confusion_matrix(predicted_labels, gt_labels)

print('accuracy %.4f' % np.mean(accuracy))
print(confusion_mat)

