"""
author: Noorvir Aulakh
date: 27 July 2017


N.B. Parts of this file are adopted from (Mahler, 2017)

Mahler, Jeffrey, Jacky Liang, Sherdil Niyaz, Michael Laskey, Richard Doan, Xinyu Liu, Juan Aparicio Ojea, and
Ken Goldberg. "Dex-Net 2.0: Deep Learning to Plan Robust Grasps with Synthetic Point Clouds and Analytic Grasp Metrics."
Grasp Metrics." Robotics, Science, and Systems, 2017. Cambridge, MA.

"""
from grasp_ucl.database.transformations import ImageTransform
import tensorflow as tf
import numpy as np
import pickle as pkl
import logging
import time
import os


# Display logging info
logging.getLogger().setLevel(logging.INFO)


class DataLoader(object):
    """ Load data from files"""

    def __init__(self, network, config):

        self._network = network
        self.config = config

        # initialise buffers to store data from loaded file
        self.img_shape = [0, self._network.img_width, self._network.img_height, self._network.img_channels]
        self.label_shape = [0, self._network.num_classes]
        self.img_data_buffer = np.empty(self.img_shape)
        self.label_data_buffer = np.empty(self.label_shape)

        # val buffer TODO: merge these with img_data_buffer
        self.img_val_data_buffer = np.empty(self.img_shape)
        self.label_val_data_buffer = np.empty(self.label_shape)

        # init methods
        self._setup_config()
        self._setup_filenames()


    def _setup_config(self):

        # data-set config
        self.filename_templates = self.config['filename_templates']

        # network config
        self.train_fraction = self._network.config['train_fraction']
        self.val_fraction = self._network.config['val_fraction']
        self.test_fraction = self._network.config['test_fraction']
        self.data_used_fraction = self._network.config['data_used_fraction']
        self.images_per_file = self.config['images_per_file']


    def _setup_filenames(self):
        """ Load file names from data-set and randomly select a pre-set fraction to train on """

        # read all data filenames
        all_filenames = os.listdir(self._network.dataset_dir)

        # get image files
        self.img_filenames = [f for f in all_filenames if f.find(self.filename_templates['depth_imgs']) > -1]
        self.label_filenames = [f for f in all_filenames if f.find(self.filename_templates['labels']) > -1]

        self.img_filenames.sort(key=lambda x: int(x[-9:-4]))
        self.label_filenames.sort(key=lambda x: int(x[-9:-4]))

        # randomly choose files for training and validation
        num_files = len(self.img_filenames)
        num_files_used = int(self.data_used_fraction * num_files)

        if num_files_used == 0:
            raise ValueError('Number of files used must be more than Zero.')

        num_train_files = int(num_files_used * self.train_fraction)
        num_val_files = int(num_files_used * self.val_fraction)
        num_test_files = int(num_files_used * self.test_fraction)

        # get total number of training datapoints
        num_datapoints = self.images_per_file * num_files_used
        self.num_train = int(self.train_fraction * num_datapoints)

        train_id_end = num_train_files
        val_id_end = train_id_end + num_val_files
        test_id_end = val_id_end + num_test_files

        # image filenames
        self.train_img_filenames = self.img_filenames[: train_id_end]
        self.val_img_filenames = self.img_filenames[train_id_end: val_id_end]
        self.train_img_filenames = self.img_filenames[val_id_end: test_id_end]

        # label filenames
        self.train_label_filenames = self.label_filenames[: train_id_end]
        self.val_label_filenames = self.label_filenames[train_id_end: val_id_end]
        self.train_label_filenames = self.label_filenames[val_id_end: test_id_end]


    def get_train_data(self):
        """ Set-up operations to create batches"""

        with tf.name_scope('train_data_queue'):
            img_data, label_data = self._network.train_data_queue.dequeue_many(n=self._network.num_train_data_dequeue, name='dequeue_op')

        with tf.name_scope('train_batch_loader'):
            if self._network.num_classes == 2:
                pos_data_idx = tf.where(tf.equal(label_data[:, 1], 1))[:, 0]
                neg_data_idx = tf.where(tf.equal(label_data[:, 0], 1))[:, 0]
                # get number of negative indices to ensure desired pos-neg split
                num_neg_multipler = (1.0 - self._network.pos_train_frac) / self._network.pos_train_frac
            else:
                pos_data_idx = tf.where(tf.not_equal(label_data[:, 0], 1))[:, 0]
                neg_data_idx = tf.where(tf.equal(label_data[:, 0], 1))[:, 0]
                # ensure equal number of datapoints from each class
                num_neg_multipler = 1/float(self._network.num_classes)

            # get positive data-points
            pos_img_data = tf.gather(img_data, pos_data_idx)
            pos_label_data = tf.gather(label_data, pos_data_idx)

            neg_img_data = tf.gather(img_data, neg_data_idx)
            neg_label_data = tf.gather(label_data, neg_data_idx)

            num_neg = tf.cast(num_neg_multipler * tf.cast(tf.shape(pos_img_data)[0], dtype=tf.float32), dtype=tf.int32)

            # get indices of negative data-points
            num_neg = tf.minimum(num_neg, tf.shape(neg_data_idx)[0])
            num_neg = tf.maximum(num_neg, 1)                        # TODO: really ugly hack for when num_neg =0 . change this
            neg_data_idx = tf.random_uniform([num_neg], 0, tf.shape(neg_data_idx)[0], dtype=tf.int32)

            # get negative indices
            neg_img_data = tf.gather(neg_img_data, neg_data_idx)
            neg_label_data = tf.gather(neg_label_data, neg_data_idx)

            imgs = tf.concat([pos_img_data, neg_img_data], axis=0)
            labels = tf.concat([pos_label_data, neg_label_data], axis=0)

            # shuffle data
            num_data_points = tf.shape(imgs)[0]
            data_idx = tf.random_shuffle(tf.range(0, num_data_points, delta=1))

            imgs = tf.gather(imgs, data_idx)
            labels = tf.gather(labels, data_idx)

        return imgs, labels


    def get_val_data(self):
        """ Dequeue data from validation buffer"""
        with tf.name_scope('val_data_queue'):
            img_data, label_data = self._network.val_data_queue.dequeue_many(n=self._network.num_val_data_dequeue, name='dequeue_op')

        with tf.name_scope('val_batch_loader'):
            if self._network.num_classes == 2:
                pos_data_idx = tf.where(tf.equal(label_data[:, 1], 1))[:, 0]
                neg_data_idx = tf.where(tf.equal(label_data[:, 0], 1))[:, 0]
                # get number of negative indices to ensure desired pos-neg split
                num_neg_multipler = (1.0 - self._network.pos_train_frac) / self._network.pos_train_frac
            else:
                pos_data_idx = tf.where(tf.not_equal(label_data[:, 0], 1))[:, 0]
                neg_data_idx = tf.where(tf.equal(label_data[:, 0], 1))[:, 0]
                # ensure equal number of datapoints from each class
                num_neg_multipler = 1/float(self._network.num_classes)

            # get positive data-points
            pos_img_data = tf.gather(img_data, pos_data_idx)
            pos_label_data = tf.gather(label_data, pos_data_idx)

            neg_img_data = tf.gather(img_data, neg_data_idx)
            neg_label_data = tf.gather(label_data, neg_data_idx)

            num_neg = tf.cast(num_neg_multipler * tf.cast(tf.shape(pos_img_data)[0], dtype=tf.float32), dtype=tf.int32)

            # get indices of negative data-points
            num_neg = tf.minimum(num_neg, tf.shape(neg_data_idx)[0])
            num_neg = tf.maximum(num_neg, 1)                        # TODO: really ugly hack for when num_neg =0 . change this
            neg_data_idx = tf.random_uniform([num_neg], 0, tf.shape(neg_data_idx)[0], dtype=tf.int32)

            # get negative indices
            neg_img_data = tf.gather(neg_img_data, neg_data_idx)
            neg_label_data = tf.gather(neg_label_data, neg_data_idx)

            imgs = tf.concat([pos_img_data, neg_img_data], axis=0)
            labels = tf.concat([pos_label_data, neg_label_data], axis=0)

            # shuffle data
            num_data_points = tf.shape(imgs)[0]
            data_idx = tf.random_shuffle(tf.range(0, num_data_points, delta=1))

            imgs = tf.gather(imgs, data_idx)
            labels = tf.gather(labels, data_idx)

        return imgs, labels


    def load_train_data(self):
        """ Load data from numpy files"""

        # get filename to load
        file_id = np.random.choice(np.shape(self.train_img_filenames)[0], size=1)
        img_filename = self.train_img_filenames[file_id]
        label_filename = self.train_label_filenames[file_id]

        # load data files
        imgs = np.load(os.path.join(self._network.dataset_dir, img_filename))['arr_0']
        labels = np.load(os.path.join(self._network.dataset_dir, label_filename))['arr_0']

        imgs = np.repeat(imgs, self._network.img_channels, axis=self._network.img_channels)

        return [imgs.astype(np.float32), labels.astype(np.float32)]


    def load_val_data(self):
        """ Load data from numpy files"""

        # get filename to load
        file_id = np.random.choice(np.shape(self.val_img_filenames)[0], size=1)
        img_filename = self.val_img_filenames[file_id]
        label_filename = self.val_label_filenames[file_id]

        # load data files
        imgs = np.load(os.path.join(self._network.dataset_dir, img_filename))['arr_0']
        labels = np.load(os.path.join(self._network.dataset_dir, label_filename))['arr_0']

        # copy 1st channel into 2nd and 3rd
        imgs = np.repeat(imgs, self._network.img_channels, axis=self._network.img_channels)

        return [imgs.astype(np.float32), labels.astype(np.float32)]


    ###### DEBUG CODE ######

    def debug_load_and_enqueue(self):

        img_path = os.path.join(self._network.cache_dir, 'debug_imgs.npz')
        label_path = os.path.join(self._network.cache_dir, 'debug_labels.npz')

        num_data = 10000

        while True:
            time.sleep(0.001)

            if os.path.exists(img_path) and os.path.exists(label_path) and not hasattr(self, 'debug_imgs'):

                self.debug_loaded = True
                debug_img_arr = np.load(img_path)['arr_0']
                label_arr = np.load(label_path)['arr_0']

                debug_idx = np.random.choice(xrange(np.shape(debug_img_arr)[0]), num_data, replace=False)
                self.debug_imgs = debug_img_arr[debug_idx]
                self.debug_labels = label_arr[debug_idx]

            elif not hasattr(self, 'debug_imgs'):

                filename_idx = xrange(np.shape(self.img_filenames)[0])

                debug_img_arr = np.empty([0, 227, 227, 1])
                debug_label_arr = np.empty([0, 2])
                num_data_points = 0

                while num_data_points < num_data:

                    file_id = np.random.choice(filename_idx, size=1)

                    imgs = np.load(os.path.join(self._network.dataset_dir, self.img_filenames[file_id]))['arr_0']
                    labels = np.load(os.path.join(self._network.dataset_dir, self.label_filenames[file_id]))['arr_0']

                    # get indices of positive examples
                    pos_indx = np.where(labels[:, 1] > 0)[0]
                    num_pos = np.shape(pos_indx)[0]

                    # get positive examples
                    pos_imgs = imgs[pos_indx]
                    pos_labels = labels[pos_indx]

                    # get all negative examples
                    neg_imgs = np.delete(imgs, pos_indx, axis=0)
                    neg_labels = np.delete(labels, pos_indx, axis=0)

                    num_neg = np.shape(neg_imgs)[0]
                    if num_neg >= num_pos:
                        neg_idx = np.random.choice(xrange(num_neg), num_pos, replace=False)
                    else:
                        neg_idx = np.random.choice(xrange(num_neg), num_neg, replace=False)

                    # get as many negative examples as positive
                    neg_imgs = neg_imgs[neg_idx]
                    neg_labels = neg_labels[neg_idx]

                    num_neg = np.shape(neg_imgs)[0]

                    debug_img_arr = np.append(debug_img_arr, pos_imgs, axis=0)
                    debug_img_arr = np.append(debug_img_arr, neg_imgs, axis=0)

                    debug_label_arr = np.append(debug_label_arr, pos_labels, axis=0)
                    debug_label_arr = np.append(debug_label_arr, neg_labels, axis=0)

                    num_data_points += (num_pos + num_neg)

                data_idx = range(np.shape(debug_img_arr)[0])
                np.random.shuffle(data_idx)

                debug_img_arr = debug_img_arr[data_idx]
                debug_label_arr = debug_label_arr[data_idx]

                print('Saving debug images')
                np.savez(img_path[:-4], debug_img_arr)
                np.savez(label_path[:-4], debug_label_arr)


                debug_idx = np.random.choice(xrange(np.shape(debug_img_arr)[0]), num_data)

                self.debug_imgs = debug_img_arr[debug_idx]
                self.debug_labels = debug_label_arr[debug_idx]

            num_imgs = np.shape(self.debug_imgs)[0]
            batch_idx = np.random.choice(xrange(num_imgs), self._network.batch_size, replace=True)

            img_batch = self.debug_imgs[batch_idx]
            # img_batch = img_batch - self._network.img_mean
            img_batch = np.repeat(img_batch, self._network.img_channels, axis=self._network.img_channels)

            label_batch = self.debug_labels[batch_idx]

            return [img_batch.astype(np.float32), label_batch.astype(np.float32)]


