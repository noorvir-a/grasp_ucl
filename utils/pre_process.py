"""
author: Noorvir Aulakh
date: 27 July 2017


N.B. Parts of this file are adopted from (Mahler, 2017)

Mahler, Jeffrey, Jacky Liang, Sherdil Niyaz, Michael Laskey, Richard Doan, Xinyu Liu, Juan Aparicio Ojea, and
Ken Goldberg. "Dex-Net 2.0: Deep Learning to Plan Robust Grasps with Synthetic Point Clouds and Analytic Grasp Metrics."
Grasp Metrics." Robotics, Science, and Systems, 2017. Cambridge, MA.

"""

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
        self.train_fraction = self._network.config['train_frac']
        self.data_used_fraction = self._network.config['data_used_fraction']
        self.images_per_file = self.config['images_per_file']

    def _setup_filenames(self):
        """ Load file names from data-set and randomly select a pre-set fraction to train on """

        # filename for index map
        train_map_name = 'train_indices_map_' + self._network.dataset_name + '_{:.5f}'.format(self.data_used_fraction).replace('.', '_') + '.pkl'
        val_map_name = 'val_indices_map_' + self._network.dataset_name + '_{:.5f}'.format(self.data_used_fraction).replace('.', '_') + '.pkl'
        train_map_path = os.path.join(self._network.cache_dir, train_map_name)
        self.val_map_path = os.path.join(self._network.cache_dir, val_map_name)

        # make a map of the train and test indices for each file or load precomputed map
        if os.path.exists(train_map_path):

            train_data = pkl.load(open(train_map_path, 'r'))
            val_data = pkl.load(open(self.val_map_path, 'r'))

            self.img_filenames = train_data['img_filenames']
            self.label_filenames = train_data['label_filenames']
            self.num_train = train_data['num_train']

            # filename for index map
            self.train_index_map = train_data['train_index_map']
            self.val_index_map = val_data

        else:
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
            filename_indices = np.random.choice(num_files, size=num_files_used, replace=False)
            filename_indices.sort()

            self.img_filenames = [self.img_filenames[k] for k in filename_indices]
            self.label_filenames = [self.label_filenames[k] for k in filename_indices]

            # get total number of training datapoints
            num_datapoints = self.images_per_file * num_files_used
            self.num_train = int(self.train_fraction * num_datapoints)

            # get training and validation indices
            all_indices = np.arange(num_datapoints)
            np.random.shuffle(all_indices)
            train_indices = np.sort(all_indices[:self.num_train])
            val_indices = np.sort(all_indices[self.num_train:])

            train_data = {}
            self.train_index_map = {}
            self.val_index_map = {}

            logging.info(self._network.get_date_time() + ' : Computing filename-to-training data indices.')

            for i, img_filename in enumerate(self.img_filenames):
                lower = i * self.images_per_file
                upper = (i + 1) * self.images_per_file
                im_arr = np.load(os.path.join(self._network.dataset_dir, img_filename))['arr_0']
                self.train_index_map[img_filename] = train_indices[(train_indices >= lower) & (train_indices < upper) &
                                                                   (train_indices - lower < im_arr.shape[0])] - lower
                self.val_index_map[img_filename] = val_indices[(val_indices >= lower) & (val_indices < upper) & (
                                                                    val_indices - lower < im_arr.shape[0])] - lower

            train_data['img_filenames'] = self.img_filenames
            train_data['label_filenames'] = self.label_filenames
            train_data['num_train'] = self.num_train
            train_data['train_index_map'] = self.train_index_map

            val_data = self.val_index_map

            logging.info(self._network.get_date_time() + ' : Writing filename-to-training map to file.')
            pkl.dump(train_data, open(train_map_path, 'w'))
            pkl.dump(val_data, open(self.val_map_path, 'w'))


    def get_train_data(self):
        """ Set-up operations to create batches"""

        with tf.name_scope('train_data_queue'):
            img_data, label_data = self._network.train_data_queue.dequeue_many(n=self._network.num_train_data_dequeue, name='dequeue_op')

        with tf.name_scope('train_batch_loader'):
            pos_data_idx = tf.where(label_data[:, 1] > 0)[:, 0]
            neg_data_idx = tf.where(label_data[:, 0] > 0)[:, 0]

            # get positive data-points
            pos_img_data = tf.gather(img_data, pos_data_idx)
            pos_label_data = tf.gather(label_data, pos_data_idx)

            neg_img_data = tf.gather(img_data, neg_data_idx)
            neg_label_data = tf.gather(label_data, neg_data_idx)

            # get number of negative indices to ensure desired pos-neg split
            num_neg_multipler = (1.0 - self._network.pos_train_frac)/self._network.pos_train_frac
            num_neg = tf.cast(num_neg_multipler * tf.cast(tf.shape(pos_img_data)[0], dtype=tf.float32), dtype=tf.int32)

            # get indices of negative data-points
            num_neg = tf.minimum(num_neg, tf.shape(neg_data_idx)[0])
            num_neg = tf.maximum(num_neg, 1)                        # TODO: really ugly hack for when num_neg =0 . change this
            neg_data_idx = tf.random_uniform([num_neg], 0, tf.shape(neg_data_idx)[0],  dtype=tf.int32)

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
        """ """
        with tf.name_scope('val_data_queue'):
            img_data, label_data = self._network.val_data_queue.dequeue_many(n=self._network.num_val_data_dequeue, name='dequeue_op')

        with tf.name_scope('val_batch_loader'):
            pos_data_idx = tf.where(label_data[:, 1] > 0)[:, 0]
            neg_data_idx = tf.where(label_data[:, 0] > 0)[:, 0]

            # get positive data-points
            pos_img_data = tf.gather(img_data, pos_data_idx)
            pos_label_data = tf.gather(label_data, pos_data_idx)

            neg_img_data = tf.gather(img_data, neg_data_idx)
            neg_label_data = tf.gather(label_data, neg_data_idx)

            # get number of negative indices to ensure desired pos-neg split
            num_neg_multipler = (1.0 - self._network.pos_train_frac)/self._network.pos_train_frac
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
        file_id = np.random.choice(np.shape(self.img_filenames)[0], size=1)
        img_filename = self.img_filenames[file_id]
        label_filename = self.label_filenames[file_id]

        # load data files
        imgs = np.load(os.path.join(self._network.dataset_dir, img_filename))['arr_0']
        labels = np.load(os.path.join(self._network.dataset_dir, label_filename))['arr_0']

        # get data-point indices assigned for training and validation
        train_idx = self.train_index_map[img_filename]
        np.random.shuffle(train_idx)

        train_idx = train_idx[:200]

        # get training data-points
        train_imgs = imgs[train_idx]
        train_labels = labels[train_idx]

        # copy 1st channel into 2nd and 3rd
        train_imgs = np.repeat(train_imgs, self._network.img_channels, axis=self._network.img_channels)

        return [train_imgs.astype(np.float32), train_labels.astype(np.float32)]


    def load_val_data(self):
        """ Load data from numpy files"""

        # get filename to load
        file_id = np.random.choice(np.shape(self.img_filenames)[0], size=1)
        img_filename = self.img_filenames[file_id]
        label_filename = self.label_filenames[file_id]

        # load data files
        imgs = np.load(os.path.join(self._network.dataset_dir, img_filename))['arr_0']
        labels = np.load(os.path.join(self._network.dataset_dir, label_filename))['arr_0']

        # get data-point indices assigned for training and validation
        val_idx = self.val_index_map[img_filename]
        np.random.shuffle(val_idx)

        # get validation data-points
        val_imgs = imgs[val_idx]
        val_labels = labels[val_idx]

        # copy 1st channel into 2nd and 3rd
        val_imgs = np.repeat(val_imgs, self._network.img_channels, axis=self._network.img_channels)

        return [val_imgs.astype(np.float32), val_labels.astype(np.float32)]


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

                np.savez(img_path, debug_img_arr)
                np.savez(label_path, debug_label_arr)


                debug_idx = np.random.choice(xrange(np.shape(debug_img_arr)[0]), num_data)

                self.debug_imgs = debug_img_arr[debug_idx]
                self.debug_labels = debug_label_arr[debug_idx]

            num_imgs = np.shape(self.debug_imgs)[0]
            batch_idx = np.random.choice(xrange(num_imgs), self._network.batch_size, replace=False)

            img_batch = self.debug_imgs[batch_idx]
            # img_batch = img_batch - self._network.img_mean
            img_batch = np.repeat(img_batch, self._network.img_channels, axis=self._network.img_channels)

            label_batch = self.debug_labels[batch_idx]

            self._network.sess.run(self._network.enqueue_op, feed_dict={self._network.img_queue_batch: img_batch,
                                                                        self._network.label_queue_batch: label_batch})


