"""
author: Noorvir Aulakh
date: 27 July 2017


N.B. Parts of this file are adopted from (Mahler, 2017)

Mahler, Jeffrey, Jacky Liang, Sherdil Niyaz, Michael Laskey, Richard Doan, Xinyu Liu, Juan Aparicio Ojea, and
Ken Goldberg. "Dex-Net 2.0: Deep Learning to Plan Robust Grasps with Synthetic Point Clouds and Analytic Grasp Metrics."
Grasp Metrics." Robotics, Science, and Systems, 2017. Cambridge, MA.

"""


from skimage import io
import skimage.transform
import numpy as np
import pickle as pkl
import logging
import os


class DataLoader(object):
    """ Load data from files"""

    def __init__(self, network, config):

        self._network = network
        self.config = config
        self.img_data_buffer = np.array([])
        self.label_data_buffer = np.array([])

        # init methods
        self._setup_config()
        self._setup_filenames()


    def _setup_config(self):

        # data-set config
        self.filename_templates = self.config[self._network.name]['filename_templates']

        # network config
        self.train_fraction = self._network.config['train_frac']
        self.images_per_file = self._network.config['images_per_file']

    def _setup_filenames(self):
        """ Load file names from data-set and randomly select a pre-set fraction to train on """

        # read all data filenames
        all_filenames = os.listdir(self._network.data_dir)

        # get image files
        self.img_filenames = [f for f in all_filenames if f.find(self.filename_templates.depth_imgs) > -1]
        # get pose files
        # self.pose_filenames = [f for f in all_filenames if f.find(ImageFileTemplates.hand_poses_template) > -1]
        # get label files
        self.label_filenames = [f for f in all_filenames if f.find(self.filename_templates.labels) > -1]

        self.img_filenames.sort(key=lambda x: int(x[-9:-4]))
        # self.pose_filenames.sort(key=lambda x: int(x[-9:-4]))
        self.label_filenames.sort(key=lambda x: int(x[-9:-4]))

        # randomly choose files for training and validation
        num_files = len(self.img_filenames)
        num_files_used = int(self.train_fraction * num_files)
        filename_indices = np.random.choice(num_files, size=num_files_used, replace=False)
        filename_indices.sort()

        self.img_filenames = [self.img_filenames[k] for k in filename_indices]
        # self.pose_filenames = [self.pose_filenames[k] for k in filename_indices]
        self.label_filenames = [self.label_filenames[k] for k in filename_indices]

        # create copy of image filenames to allow for parallel accessing
        self.img_filenames_copy = self.img_filenames[:]

        # shuffle data and create map from indices to datapoints

        # get total number of training datapoints
        num_datapoints = self.images_per_file * num_files
        self.num_train = int(self.train_fraction * num_datapoints)

        # get training and validation indices
        all_indices = np.arange(num_datapoints)
        np.random.shuffle(all_indices)
        train_indices = np.sort(all_indices[:self.num_train])
        val_indices = np.sort(all_indices[self.num_train:])

        # make a map of the train and test indices for each file
        logging.info(self._network.get_date_time() + ' : Computing filename-to-training data indices...')
        train_index_map_filename = os.path.join(self._network.cache_dir, 'train_indices_map.pkl')
        self.val_index_map_filename = os.path.join(self._network.cache_dir, 'val_indices_map.pkl')


        if os.path.exists(train_index_map_filename):
            self.train_index_map = pkl.load(open(train_index_map_filename, 'r'))
            self.val_index_map = pkl.load(open(self.val_index_map_filename, 'r'))

        else:
            self.train_index_map = {}
            self.val_index_map = {}

            for i, img_filename in enumerate(self.img_filenames):
                lower = i * self.images_per_file
                upper = (i + 1) * self.images_per_file
                im_arr = np.load(os.path.join(self._network.data_dir, img_filename))['arr_0']
                self.train_index_map[img_filename] = train_indices[(train_indices >= lower) & (train_indices < upper) &
                                                                   (train_indices - lower < im_arr.shape[0])] - lower
                self.val_index_map[img_filename] = val_indices[(val_indices >= lower) & (val_indices < upper) & (
                                                                    val_indices - lower < im_arr.shape[0])] - lower
            pkl.dump(self.train_index_map, open(train_index_map_filename, 'w'))
            pkl.dump(self.val_index_map, open(self.val_index_map_filename, 'w'))


    def get_next_batch(self):
        """ Get the next batch of training data"""

        # get next batch from loaded array. if the array length is smaller than batch size, load next file and append to array
        if len(self.img_data_buffer) < self._network.batch_size:
            # load next file

            file_id = np.random.choice(len(self.img_filenames), size=1)[0]
            img_filename = self.img_filenames[file_id]
            label_filename = self.label_filenames[file_id]

            # create file-paths
            img_file_path = os.path.join(self._network.data_dir, img_filename)
            label_file_path = os.path.join(self._network.data_dir, label_filename)

            # load new data
            img_data = np.load(img_file_path)['arr_0']
            label_data = np.load(label_file_path)['arr_0']

            # get data-point indices assigned for training
            train_idx = self.train_index_map[img_filename]
            np.random.shuffle(train_idx)

            # remove validation data-points
            img_data = img_data[train_idx]
            label_data = label_data[train_idx]

            # add new data to buffers
            self.img_data_buffer = np.concatenate(self.img_data_buffer, img_data)
            self.label_data_buffer = np.concatenate(self.label_data_buffer, label_data)

        img_batch = self.img_data_buffer[:self._network.batch_size]
        label_batch = self.label_data_buffer[:self._network.batch_size]

        # remove batch from buffers
        self.img_data_buffer = np.delete(self.img_data_buffer, np.s_[:self._network.batch_size])
        self.label_data_buffer = np.delete(self.label_data_buffer, np.s_[:self._network.batch_size])

        return img_batch, label_batch


    def load_guant_batch(self):
        """ Load the next training/test batch"""

        img_batch = 0
        labels_batch = 0

        return img_batch, labels_batch

# class ImageOps():
#
# def upsample_skimage(factor, input_img):
#     # Pad with 0 values, similar to how Tensorflow does it.
#     # Order=1 is bilinear upsampling
#     return skimage.transform.rescale(input_img,
#                                      factor,
#                                      mode='constant',
#                                      cval=0,
#                                      order=1)
#
#
# dir = '/home/noorvir/datasets/gqcnn/dexnet_mini/'
# file = 'depth_ims_tf_00000.npz'
#
# img = np.load(dir + file)['arr_0'][0]
# img = np.reshape(img, [32, 32])
#
# # io.imshow(img, interpolation='none')
# # io.show()
# upsampled_img_skimage = upsample_skimage(factor=8, input_img=img)
# io.imshow(upsampled_img_skimage, interpolation='none')
# io.show()
# pass