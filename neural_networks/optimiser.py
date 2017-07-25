"""
author: Noorvir Aulakh
date: 18 July 2017

N.B. Parts of this file are adopted from (Mahler, 2017)

Mahler, Jeffrey, Jacky Liang, Sherdil Niyaz, Michael Laskey, Richard Doan, Xinyu Liu, Juan Aparicio Ojea, and
Ken Goldberg. "Dex-Net 2.0: Deep Learning to Plan Robust Grasps with Synthetic Point Clouds and Analytic Grasp Metrics."
Grasp Metrics." Robotics, Science, and Systems, 2017. Cambridge, MA.

"""

import argparse
import copy
import cv2
import json
import logging
import numbers
import numpy as np
import cPickle as pkl
import os
import random
import scipy.misc as sm
import scipy.ndimage.filters as sf
import scipy.ndimage.morphology as snm
import scipy.stats as ss
import skimage.draw as sd
import signal
import sys
import shutil
import threading
import time
import urllib
import matplotlib.pyplot as plt
import tensorflow as tf
import yaml
from autolab_core import YamlConfig
import autolab_core.utils as utils
import collections

import IPython

from gqcnn.learning_analysis import ClassificationResult, RegressionResult
from gqcnn.optimizer_constants import ImageMode, TrainingMode, PreprocMode, InputDataMode, GeneralConstants, \
    ImageFileTemplates
from gqcnn.train_stats_logger import TrainStatsLogger


class SGDOptimizer(object):
    """ Optimizer for gqcnn object """

    def __init__(self, network, config):

        self.network = network
        self.config = {'data_dir': r'/home/noorvir/datasets/gqcnn/dexnet_mini',
                       'temp_data_dir': r'/home/noorvir/datasets/gqcnn/temp_mini',
                       'images_in_file': 1000,
                       'optimiser': 'momentum',
                       'momentum_rate': 0.9,
                       'l2_regulariser': 0.95,
                       'batch_size_train': 1000,
                       'gq_train_fraction': 0.5,
                       'num_epochs': 20,
                       'img_height': 32,
                       'img_width': 32,
                       'num_channels': 1,
                       'log_frequency': 50,
                       'eval_frequency': 1000,
                       'save_frequency': 1000,
                       'target_metric_name': 'robust_ferrari_canny',
                       'train_fraction': 0.8,
                       'decay_step_multiplier': 0.6,
                       'queue_capacity': 100,
                       'drop_fc3': 0,
                       'drop_fc4': 0,
                       'fc3_drop_rate': 0.5,
                       'fc4_drop_rate': 0.5}

        # training config
        # self.num_train =

        self.dim_pose = 1
        # base learning rate to use for exponential learning rate decay
        self.base_learning_rate = self.learning_rate = 0.001
        self.decay_rate = 0.95
        # self.decay_step_multiplier =
        self.decay_step = 150000
        # self.decay_step = self.decay_step_multiplier * self.num_train



    def _read_config(self):
        """ Initialise variables from config """

        self.data_dir = self.config['data_dir']
        self.temp_data_dir = self.config['temp_data_dir']
        self.images_in_file = self.config['images_in_file']
        self.momentum_rate = self.config['momentum_rate']
        self.num_epochs = self.config['num_epochs']
        self.batch_size_train = self.config['batch_size_train']
        self.l2_regulariser = self.config['l2_regulariser']
        self.img_height = self.config['img_height']
        self.img_width = self.config['img_width']
        self.num_channels = self.config['num_channels']
        self.log_frequency = self.config['log_frequency']
        self.eval_frequency = self.config['eval_frequency']
        self.save_frequency = self.config['save_frequency']
        self.target_metric_name = self.config['target_metric_name']
        self.train_fraction = self.config['train_fraction']
        self.decay_step_multiplier = self.config['decay_step_multiplier']
        self.queue_capacity = self.config['queue_capacity']

    def _setup_filenames(self):
        """ Load file names from data-set and randomly select a pre-set fraction to train on """

        self.data_dir = self.config['data_dir']

        # read all data filenames
        all_filenames = os.listdir(self.data_dir)

        # TODO: choose the type of input image to use (color, binary etc)
        # get image files
        self.img_filenames = [f for f in all_filenames if
                             f.find(ImageFileTemplates.depth_im_tf_table_tensor_template) > -1]
        # get pose files
        self.pose_filenames = [f for f in all_filenames if f.find(ImageFileTemplates.hand_poses_template) > -1]
        # get label files (robust_ferrari_canny in most cases)
        self.label_filenames = [f for f in all_filenames if f.find(self.target_metric_name) > -1]

        self.img_filenames.sort(key=lambda x: int(x[-9:-4]))
        self.pose_filenames.sort(key=lambda x: int(x[-9:-4]))
        self.label_filenames.sort(key=lambda x: int(x[-9:-4]))

        # sort filename to match inputs and labels

        # randomly choose files for training and validation
        num_files = len(self.img_filenames)
        num_files_used = int(self.train_fraction * num_files)
        filename_indices = np.random.choice(num_files, size=num_files_used, replace=False)
        filename_indices.sort()

        self.img_filenames = [self.img_filenames[k] for k in filename_indices]
        self.pose_filenames = [self.pose_filenames[k] for k in filename_indices]
        self.label_filenames = [self.label_filenames[k] for k in filename_indices]
        # shuffle them and store in a dictionary/list

        # create copy of image filenames to allow for parallel accessing
        self.img_filenames_copy = self.img_filenames[:]

        # shuffle data and create map from indices to datapoints

        ### Taken directly from (Mahler, 2017) ###

        # get total number of training datapoints and set the decay_step
        num_datapoints = self.images_in_file * num_files
        self.num_train = int(self.train_fraction * num_datapoints)
        self.decay_step = self.decay_step_multiplier * self.num_train

        # get training and validation indices
        all_indices = np.arange(num_datapoints)
        np.random.shuffle(all_indices)
        train_indices = np.sort(all_indices[:self.num_train])
        val_indices = np.sort(all_indices[self.num_train:])

        # make a map of the train and test indices for each file
        logging.info('Computing indices image-wise')
        train_index_map_filename = os.path.join(self.temp_data_dir, 'train_indices_image_wise.pkl')
        self.val_index_map_filename = os.path.join(self.temp_data_dir, 'val_indices_image_wise.pkl')
        if os.path.exists(train_index_map_filename):
            self.train_index_map = pkl.load(open(train_index_map_filename, 'r'))
            self.val_index_map = pkl.load(open(self.val_index_map_filename, 'r'))
        else:
            self.train_index_map = {}
            self.val_index_map = {}
            for i, img_filename in enumerate(self.img_filenames):
                lower = i * self.images_in_file
                upper = (i + 1) * self.images_in_file
                im_arr = np.load(os.path.join(self.data_dir, img_filename))['arr_0']
                self.train_index_map[img_filename] = train_indices[(train_indices >= lower) & (train_indices < upper) &
                                                                   (train_indices - lower < im_arr.shape[0])] - lower
                self.val_index_map[img_filename] = val_indices[(val_indices >= lower) & (val_indices < upper) & (
                                                                    val_indices - lower < im_arr.shape[0])] - lower
            pkl.dump(self.train_index_map, open(train_index_map_filename, 'w'))
            pkl.dump(self.val_index_map, open(self.val_index_map_filename, 'w'))


    def _setup_training_data(self):
        """ Setup training batches and queues for loading data"""

        with tf.name_scope('train_data_node'):
            self.img_train_batch = tf.placeholder(tf.float32, (self.batch_size_train, self.img_height, self.img_width,
                                                               self.num_channels))

        with tf.name_scope('train_pose_node'):
            self.pose_train_batch = tf.placeholder(tf.float32, (self.batch_size_train, self.dim_pose))

        with tf.name_scope('train_labels_node'):
            self.train_labels_batch = tf.placeholder(tf.int64, (self.batch_size_train,))

        # load images, poses and labels
        # self.train_img_data = np.load(os.path.join(self.data_dir, self.img_filenames[0]))['arr_0']
        # self.pose_data = np.load(os.path.join(self.data_dir, self.pose_filenames[0]))['arr_0']
        # self.metric_data = np.load(os.path.join(self.data_dir, self.label_filenames[0]))['arr_0']

        # use tf.FIFO to set up queue and run enqueue  and dequeue operations
        with tf.name_scope('data_queue'):
            self.q = tf.FIFOQueue(self.queue_capacity, [tf.float32, tf.float32, tf.int64],
                                  shapes=[(self.batch_size_train, self.img_height, self.img_width, self.num_channels),
                                          (self.batch_size_train, self.dim_pose), (self.batch_size_train,)])
            self.enqueue_op = self.q.enqueue([self.img_train_batch, self.pose_train_batch, self.train_labels_batch])
            self.train_labels_node = tf.placeholder(tf.int64, (self.batch_size_train,))
            self.input_img_node, self.input_pose_node, self.train_labels_node = self.q.dequeue()


    def _create_optimizer(self, loss, batch_num, var_list):
        """

        :param loss:
            loss to use, generated with _create_loss()
        :param batch_num:
            variable to keep track of the current gradient step number
        :param var_list:
            list of tf.Variable objects to update to minimize loss(ex. network weights)

        :return:
            optimiser
        """

        if self.config['optimiser'] == 'momentum':
            return tf.train.MomentumOptimizer(self.learning_rate, self.momentum_rate).minimize(loss,
                                                                                               global_step=batch_num,
                                                                                               var_list=var_list)
        elif self.config['optimiser'] == 'adam':
            return tf.train.AdamOptimizer(self.learning_rate).minimize(loss,
                                                                       global_step=batch_num,
                                                                       var_list=var_list)
        elif self.config['optimiser'] == 'rmsprop':
            return tf.train.RMSPropOptimizer(self.learning_rate).minimize(loss,
                                                                          global_step=batch_num,
                                                                          var_list=var_list)
        else:
            raise ValueError('Optimiser %s not supported' % (self.config['optimiser']))


    def _create_losses(self, loss_type):
        """

        :param loss_type:
        :return:
        """

        if loss_type == 'l2':
            return tf.nn.l2_loss(tf.subtract(self.network_output, self.train_labels_node))
        elif loss_type == 'sparse':
            return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(_sentinel=None,
                                                                                 labels=self.train_labels_node,
                                                                                 logits=self.network_output,
                                                                                 name=None))
        elif loss_type == 'xentropy':
            return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(_sentinel=None,
                                                                          labels=self.train_labels_node,
                                                                          logits=self.network_output,
                                                                          name=None))
        else:
            raise ValueError('Loss "%s" not supported' % loss_type)


    def optimise(self):
        """ Setup network and optimise"""

        # initialise variables and placeholders
        batch_num = tf.Variable(0)

        # setup config variables
        self._read_config()

        self.network.init_weights_gaussian()
        self.weights = self.network.get_weights()

        # load and randomise training-data files
        self._setup_filenames()

        # load training data
        self._setup_training_data()

        # 1. setup network
        drop_fc3 = False
        if 'drop_fc3' in self.config.keys() and self.config['drop_fc3']:
            drop_fc3 = True
        drop_fc4 = False
        if 'drop_fc4' in self.config.keys() and self.config['drop_fc4']:
            drop_fc4 = True

        fc3_drop_rate = self.config['fc3_drop_rate']
        fc4_drop_rate = self.config['fc4_drop_rate']

        # build training and validation networks
        with tf.name_scope('validation_network'):
            self.network.initialize_network()  # builds validation network inside gqcnn class
        with tf.name_scope('training_network'):
            self.network_output = self.network._build_network(self.input_img_node, self.input_pose_node, drop_fc3,
                                                              drop_fc4, fc3_drop_rate, fc4_drop_rate)

        # 2. setup losses
        with tf.name_scope('losses'):
            # grasp quality loss
            gq_loss = self._create_losses('sparse')
            # binary success metric loss
            # bin_loss = self._create_losses('binary')

        # 3. regularisation
        layer_weights = self.weights.__dict__.values()
        with tf.name_scope('regularisation'):
            regularisers = tf.nn.l2_loss(layer_weights[0])
            for w in layer_weights[1:]:
                regularisers += tf.nn.l2_loss(w)

            gq_loss += self.l2_regulariser * regularisers

        # 4. learning rate decay
        self.learning_rate = tf.train.exponential_decay(
            self.base_learning_rate,
            batch_num * self.batch_size_train,  # current index into the data-set.
            self.decay_step,
            self.decay_rate,
            staircase=True)

        # explicitly define the variables to update while minimising loss
        var_list = self.weights.__dict__.values()

        # 5. create optimiser
        with tf.name_scope('gq_optimiser'):
            self.gq_optimiser = self._create_optimizer(gq_loss, batch_num, var_list)
        # with tf.name_scope('bin_optimiser'):
        #     self.bin_optimiser = self._create_optimizer(bin_loss, batch_num, var_list)

        # open tf session
        with tf.Session() as self.sess:

            # initialise tf variables
            self.sess.run(tf.global_variables_initializer())

            # start training iterations
            # for epoch in xrange(self.num_epochs):
            for step_num in xrange(self.num_epochs * int(self.num_train/self.batch_size_train)):

                # 6. run optimiser
                _, _, loss = self.sess.run([self.gq_optimiser, self.learning_rate, gq_loss, self.train_labels_node,
                                               self.network_output, self.input_img_node, self.input_pose_node])

                # # alternate between training on quality & binary metric
                # if np.random.rand() < self.gq_train_fraction:
                #     # run grasp quality labels optimiser
                #
                # else:
                #     # run binary labels optimiser
                #     self.sess.run(self.bin_optimiser)

                print step_num, loss

                if step_num % self.log_frequency == 0:
                    # log metrics to file/screen
                    print step_num, loss

                if step_num % self.eval_frequency == 0:
                    # run validation network
                    pass

                if step_num % self.save_frequency == 0:
                    # save model and metrics
                    pass

            pass
