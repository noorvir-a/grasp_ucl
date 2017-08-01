"""
author: Noorvir Aulakh
date: 27 July 2017


N.B. Parts of this file are adopted from (Mahler, 2017)

Mahler, Jeffrey, Jacky Liang, Sherdil Niyaz, Michael Laskey, Richard Doan, Xinyu Liu, Juan Aparicio Ojea, and
Ken Goldberg. "Dex-Net 2.0: Deep Learning to Plan Robust Grasps with Synthetic Point Clouds and Analytic Grasp Metrics."
Grasp Metrics." Robotics, Science, and Systems, 2017. Cambridge, MA.

"""

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
        train_map_name = 'train_indices_map_' + self._network.dataset_name + '_{:.2f}'.format(self.data_used_fraction).replace('.', '_') + '.pkl'
        val_map_name = 'val_indices_map_' + self._network.dataset_name + '_{:.2f}'.format(self.data_used_fraction).replace('.', '_') + '.pkl'
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
            # get pose files
            # self.pose_filenames = [f for f in all_filenames if f.find(ImageFileTemplates.hand_poses_template) > -1]
            # get label files
            self.label_filenames = [f for f in all_filenames if f.find(self.filename_templates['labels']) > -1]

            self.img_filenames.sort(key=lambda x: int(x[-9:-4]))
            # self.pose_filenames.sort(key=lambda x: int(x[-9:-4]))
            self.label_filenames.sort(key=lambda x: int(x[-9:-4]))

            # randomly choose files for training and validation
            num_files = len(self.img_filenames)
            num_files_used = int(self.data_used_fraction * num_files)
            filename_indices = np.random.choice(num_files, size=num_files_used, replace=False)
            filename_indices.sort()

            self.img_filenames = [self.img_filenames[k] for k in filename_indices]
            self.label_filenames = [self.label_filenames[k] for k in filename_indices]

            # get total number of training datapoints
            num_datapoints = self.images_per_file * num_files
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


    # TODO: make one method out of this and get_next_val_batch
    def get_next_val_batch(self):
        """ Get next validation batch """

        # get next batch from loaded array. if the array length is smaller than batch size, load next file and append to array
        while len(self.img_val_data_buffer) < self._network.batch_size:
            # load next file

            file_id = np.random.choice(len(self.img_filenames), size=1)[0]
            img_filename = self.img_filenames[file_id]
            label_filename = self.label_filenames[file_id]

            # create file-paths
            img_file_path = os.path.join(self._network.dataset_dir, img_filename)
            label_file_path = os.path.join(self._network.dataset_dir, label_filename)

            # load new data
            # TODO: Convert labels to binary
            img_data = np.load(img_file_path)['arr_0']
            label_data = np.load(label_file_path)['arr_0']

            # get data-point indices assigned for training
            train_idx = self.val_index_map[img_filename]
            np.random.shuffle(train_idx)

            # remove validation data-points
            img_data = img_data[train_idx]
            label_data = label_data[train_idx]

            # copy first channel into all three (for compatibility with AlexNet)
            img_data = np.repeat(img_data, self._network.img_channels, axis=self._network.img_channels)

            # add new data to buffers
            self.img_val_data_buffer = np.concatenate((self.img_val_data_buffer, img_data), axis=0)
            self.label_val_data_buffer = np.concatenate((self.label_val_data_buffer, label_data), axis=0)

        img_batch = self.img_val_data_buffer[:self._network.batch_size]
        label_batch = self.label_val_data_buffer[:self._network.batch_size]

        # remove batch from buffers
        self.img_val_data_buffer = np.delete(self.img_val_data_buffer, np.s_[:self._network.batch_size], axis=0)
        self.label_val_data_buffer = np.delete(self.label_val_data_buffer, np.s_[:self._network.batch_size], axis=0)

        return img_batch, label_batch


    def get_next_batch(self):
        """ Get the next batch of training data"""

        # if the buffer length is smaller than batch size, load next file
        while len(self.img_data_buffer) < self._network.batch_size:
            # load next file

            file_id = np.random.choice(len(self.img_filenames), size=1)[0]
            img_filename = self.img_filenames[file_id]
            label_filename = self.label_filenames[file_id]

            # create file-paths
            img_file_path = os.path.join(self._network.dataset_dir, img_filename)
            label_file_path = os.path.join(self._network.dataset_dir, label_filename)

            # load new data
            # TODO: Convert labels to binary
            img_data = np.load(img_file_path)['arr_0']
            label_data = np.load(label_file_path)['arr_0']

            # get data-point indices assigned for training
            train_idx = self.train_index_map[img_filename]
            np.random.shuffle(train_idx)

            # remove validation data-points
            img_data = img_data[train_idx]
            label_data = label_data[train_idx]

            # copy first channel into all three (for compatibility with AlexNet)
            img_data = np.repeat(img_data, self._network.img_channels, axis=self._network.img_channels)

            # add new data to buffers
            self.img_data_buffer = np.concatenate((self.img_data_buffer, img_data), axis=0)
            self.label_data_buffer = np.concatenate((self.label_data_buffer, label_data), axis=0)

        img_batch = self.img_data_buffer[:self._network.batch_size]
        label_batch = self.label_data_buffer[:self._network.batch_size]

        # remove batch from buffers
        self.img_data_buffer = np.delete(self.img_data_buffer, np.s_[:self._network.batch_size], axis=0)
        self.label_data_buffer = np.delete(self.label_data_buffer, np.s_[:self._network.batch_size], axis=0)

        return img_batch, label_batch


    def load_and_enqueue(self):
        """ Load the next batch of data and enqueue"""

        while True:

            time.sleep(0.001)
            img_batch = np.empty(self.img_shape)
            label_batch = np.empty(self.label_shape)

            # if the buffer length is smaller than batch size, load next file
            while len(img_batch) < self._network.batch_size:
                # load next file

                file_id = np.random.choice(len(self.img_filenames), size=1)[0]
                img_filename = self.img_filenames[file_id]
                label_filename = self.label_filenames[file_id]

                # create file-paths
                img_file_path = os.path.join(self._network.dataset_dir, img_filename)
                label_file_path = os.path.join(self._network.dataset_dir, label_filename)

                # load new data
                # TODO: Convert labels to binary
                img_data = np.load(img_file_path)['arr_0']
                label_data = np.load(label_file_path)['arr_0']

                # get data-point indices assigned for training
                train_idx = self.train_index_map[img_filename]
                np.random.shuffle(train_idx)

                # remove validation data-points
                img_data = (img_data[train_idx] - self._network.img_mean)/self._network.img_stdev
                label_data = label_data[train_idx]

                # copy first channel into all three (for compatibility with AlexNet)
                img_data = np.repeat(img_data, self._network.img_channels, axis=self._network.img_channels)

                img_batch = img_data[:self._network.batch_size]
                label_batch = label_data[:self._network.batch_size]


            self._network.sess.run(self._network.enqueue_op, feed_dict={self._network.img_queue_batch: img_batch,
                                                                        self._network.label_queue_batch: label_batch})
