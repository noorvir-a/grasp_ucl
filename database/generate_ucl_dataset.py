"""
author: Noorvir Aulakh
date: 20 July 2017

N.B. Parts of this file are inspired by (Mahler, 2017)

Mahler, Jeffrey, Jacky Liang, Sherdil Niyaz, Michael Laskey, Richard Doan, Xinyu Liu, Juan Aparicio Ojea, and
Ken Goldberg. "Dex-Net 2.0: Deep Learning to Plan Robust Grasps with Synthetic Point Clouds and Analytic Grasp Metrics."
arXiv preprint arXiv:1703.09312 (2017).


------------------------------------------------------------------------------------------------------------------------

Approach Angles:
    N.B. There is some confusion with the API documentation for the two angles that define the grasp. I assume (which is
    the more likely case) that grasp_approach_angle is the angle between the vector pointing out of the gripper jaws and
    the table normal. The grasp_axis_angle is the angle between the vector between the gripper jaws and the table normal

Transformation:
    The convention for variable names for rigid transformations is T_fromframe_toframe.

"""

from __future__ import print_function
from autolab_core import YamlConfig, RigidTransform, Point
from meshpy import ObjFile, SceneObject, UniformPlanarWorksurfaceImageRandomVariable
from dexnet.database.keys import *
from perception import RenderMode
from dexnet.grasping import GraspCollisionChecker, RobotGripper
import dexnet.database.database as db

from gqcnn import Visualizer as vis2d
from gqcnn import Grasp2D


from grasp_ucl.utils.visualise import UCLVisualiser as vis
from grasp_ucl.database.transformations import ImageTransform

from natsort import natsorted, ns
import numpy as np
import cPickle as pkl
import logging
import warnings
import operator
import time
import os

# Display logging info
logging.getLogger().setLevel(logging.INFO)


class UCLDatabaseGQCNN(object):
    """
    Create custom data-set from GQCNN data-set (Mahler, 2017). Use bins (0.0, 0.2, ... 1.0) as labels to learn a grasp-
    quality function.
    """

    def __init__(self, config):

        self.config = config
        self.metric_list = []
        self.database_dir = config['database_dir']
        self.dataset_dir = config['dataset_dir']
        self.dataset_output_dir = config['dataset_output_dir']
        self.dataset_cache_dir = config['dataset_cache_dir']
        self.grasp_metric = config['grasp_metric']
        self.metric_stats_filename = config['metric_stats_filename']
        self.labels = config['labels']
        self.bin_step = config['bin_step']
        self.label_threshold = config['label_threshold']


    def get_metric_stats(self):
        """ Get the max and min values of the grasp metric for normalisation """

        filenames = os.listdir(self.dataset_dir)
        metric_filenames = [name for name in filenames if name.find(self.grasp_metric) > -1]

        st_time = time.time()
        # open and read all metric data
        for filename in metric_filenames:

            with np.load(os.path.join(self.dataset_dir, filename)) as f:
                data = f['arr_0']
                data = data[np.where(data != 0)]
                self.metric_list = np.concatenate((self.metric_list, data))

        self.metric_max = np.max(self.metric_list)
        self.metric_min = np.min(self.metric_list)

        stats_file_path = os.path.join(self.database_dir, self.grasp_metric + '_metric_stats.pkl')
        with open(stats_file_path, 'wb') as pkl_file:
            print('Writing %s stats to file: %s' % (self.grasp_metric, str(stats_file_path)))
            stats_dict = {'metric_list': self.metric_list, 'metric_max': self.metric_max, 'metric_min': self.metric_min}
            pkl.dump(stats_dict, pkl_file)

            print('Time taken for writing %d non-zero metric points: %s(s)\n' % (len(self.metric_list),
                                                                                 (time.time() - st_time)))


    def create_images(self, input_filename_template, output_filename_template):
        """ Modify existing images to the desired format"""

        # load images
        img_filenames = self.load_filenames(self.dataset_dir, input_filename_template)

        num_files = len(img_filenames)
        for _id, filename in enumerate(img_filenames):

            imgs = np.load(os.path.join(self.dataset_dir, filename))['arr_0']
            logging.info('Resampling images from file %d of %d. %d images.' % (_id, num_files, len(imgs)))

            # upsample
            scaled_images = ImageTransform.resample(imgs, self.config['output_img_size'])
            scaled_images = {'arr_0': scaled_images}

            # save
            output_filename = output_filename_template + '_' + str(filename[-9:-4])
            output_file_path = os.path.join(self.dataset_output_dir, output_filename)
            np.savez(output_file_path, scaled_images)


    def create_labels(self):
        """ Creates and saves one-hot normalised and binned grasp quality labels for each label file in GQCNN dataset"""

        if not hasattr(self, 'metric_max'):

            stats_file_path = os.path.join(self.dataset_cache_dir, self.metric_stats_filename)
            if not os.path.exists(stats_file_path):
                raise IOError('Metric statistics not found. Make sure UCLDatabaseGQCNN.get_metric_stats() first.')

            stats = pkl.load(open(stats_file_path, 'rb'))
            self.metric_max = stats['metric_max']
            self.metric_min = stats['metric_min']
            self.metric_list = stats['metric_list']
            self.num_metric_points = len(self.metric_list)
            self.num_seen_data_points = 0                   # used to compute percentile normalisation


        filenames = os.listdir(self.dataset_dir)
        metric_filenames = [name for name in filenames if name.find(self.grasp_metric) > -1]
        metric_filenames = natsorted(metric_filenames)          # sort file indices

        print('Starting label conversion.')
        st_time = time.time()

        data = np.array([])
        # load data
        print('Loading data...')
        for i, filename in enumerate(metric_filenames):
            file_data = np.load(os.path.join(self.dataset_dir, filename))['arr_0']
            data = np.concatenate((data, file_data))

            if i % 100 == 0:
                print('Loading data from file number %d out of %d' % (i + 1, len(metric_filenames)))


        # convert raw data to binary labels
        logging.info('Thresholding data data...')
        binary_data = self.create_binary(np.copy(data), threshold=self.label_threshold)
        binary_data = self.create_one_hot(binary_data, [0, 1])

        # normalise
        print('Normalising data...')
        normalised_data = self.normalise(data, normalisation_type=self.config['normalisation_type'])

        # bin
        print('Binning data...')
        binned_data = self.bin(normalised_data, bin_step=self.bin_step)

        # convert to one_hot representation
        print('Converting data to one-hot vector representation...')
        one_hot_data = self.create_one_hot(binned_data, self.labels)

        # save to files
        start_index = 0
        end_index = self.config['num_points_per_file']
        print('Startin file write...')
        for i, filename in enumerate(metric_filenames):

            # filenames
            binary_label_filename = 'binary_labels_' + filename[-9:-4]
            binary_label_path = os.path.join(self.dataset_output_dir, binary_label_filename)

            binned_label_filename = 'binned_labels_' + filename[-9:-4]
            binned_label_path = os.path.join(self.dataset_output_dir, binned_label_filename)

            one_hot_label_filename = 'one_hot_labels_' + filename[-9:-4]
            one_hot_label_path = os.path.join(self.dataset_output_dir, one_hot_label_filename)

            if end_index <= (len(binned_data) - 1):
                binary_file_data = binary_data[start_index: end_index]
                binned_file_data = binned_data[start_index: end_index]
                one_hot_file_data = one_hot_data[start_index: end_index, :]

                start_index = end_index
                end_index = end_index + self.config['num_points_per_file']

            else:
                binary_file_data = binary_data[start_index:]
                binned_file_data = binned_data[start_index:]
                one_hot_file_data = one_hot_data[start_index:, :]

            # add a extra array dimension for convenience at training time
            binned_file_data = np.expand_dims(binned_file_data, axis=1)

            # save
            np.savez_compressed(binary_label_path, binary_file_data)
            np.savez_compressed(binned_label_path, binned_file_data)
            np.savez_compressed(one_hot_label_path, one_hot_file_data)

            if i % 100 == 0:
                print('Saving file number %d out of %d' % (i + 1, len(metric_filenames)))

        print('All labels written to file in %s(s)' % str(time.time() - st_time))


    def normalise(self, data, normalisation_type='linear'):
        """ Normalise data to the range [0,1] """

        if normalisation_type == 'linear':
            normalised_data = (data - self.metric_min)/(self.metric_max - self.metric_min)
            normalised_data = np.clip(normalised_data, 0, float('inf'))         # corner case for grasps in collision

        elif normalisation_type == 'percentile':
            normalised_data = np.copy(data)
            # non-zero data list
            map_non_zero_org = np.where(data != 0)
            non_zero_data = data[map_non_zero_org]
            num_data_points = len(non_zero_data)

            # keep track of indices
            map_sorted_non_zero = np.argsort(non_zero_data)

            # normalise
            data_idx = np.arange(1, num_data_points + 1)
            sorted_normalised_data = data_idx / float(num_data_points)

            # map data back to original indices
            non_zero_normalised_data = np.zeros(num_data_points)
            non_zero_normalised_data[map_sorted_non_zero] = sorted_normalised_data
            normalised_data[map_non_zero_org] = non_zero_normalised_data

        elif normalisation_type == 'gamma':
            pass

        else:
            raise ValueError('Unknown normalisation_type %s' % normalisation_type)

        return normalised_data


    def visualise(self, vis_type='histogram'):
        """ Visualise different data metrics """

        # load data metrics from file
        if not hasattr(self, 'metric_max'):
            stats_file_path = os.path.join(self.database_dir, self.grasp_metric + '_metric_stats.pkl')
            if not os.path.exists(stats_file_path):
                raise IOError('Metric statistics not found. Make sure UCLDatabaseGQCNN.get_metric_stats() first.')

            stats = pkl.load(open(stats_file_path, 'rb'))
            self.metric_max = stats['metric_max']
            self.metric_min = stats['metric_min']
            self.metric_list = stats['metric_list']

        filenames = os.listdir(self.dataset_dir)
        metric_filenames = [name for name in filenames if name.find(self.grasp_metric) > -1]

        if vis_type == 'histogram':
            bins = {}
            # histogram properties
            num_bins = self.config['vis_histogram_num_bins']
            bin_step = self.config['bin_step']
            histogram_data = [0.0] * num_bins

            # calculate bin edges automatically if bin_step not specified
            if bin_step == 'auto':
                bins['edges'] = list(np.linspace(self.metric_min, self.metric_max, num_bins + 1))
            else:
                bins['edges'] = list(np.arange(self.metric_min - (bin_step/2), self.metric_max + bin_step, bin_step))

            bins['labels'] = list(np.linspace(self.metric_min, self.metric_max, num_bins))

            for filename in metric_filenames:
                with np.load(os.path.join(self.dataset_dir, filename)) as f:
                    data = f['arr_0']
                    # binned_data, _ = self.histogram(data, num_bins, self.metric_min, self.metric_max)
                    binned_data = np.histogram(data, bins['edges'])[0]
                    histogram_data = map(operator.add, histogram_data, binned_data)

            # visualise
            vis.histogram(histogram_data, bins['labels'])


    @staticmethod
    def histogram(data, num_bins, min_bin=0, max_bin=1):
        """ Digitise data into num_bins bins in the range [0,1] """

        bins_edges = np.linspace(min_bin, max_bin, num_bins)
        binned_data = np.histogram(data, bins_edges)
        return binned_data[0], binned_data[1]

    @staticmethod
    def bin(data, bin_step=0.2, min_bin=0):
        """ Digitise data into bins of step bin_step. Expects data normalise to the interval [0, 1] """

        # make sure data is an array
        data = np.array(data)

        return min_bin + np.round(data/bin_step) * bin_step

    @staticmethod
    def create_one_hot(data, labels):
        """ Create one hot labels from sorted list of labels"""
        # create array of indexes into one_hot labels
        label_map = {}
        for idx, label in enumerate(labels):
            label_map[label] = idx

        data_idx = np.copy(data)
        for key in label_map:
            data_idx[data == key] = label_map[key]

        # turn to one_hot
        one_hot_data = np.zeros([len(data), len(labels)])
        one_hot_data[np.arange(len(data)), data_idx.astype(int)] = 1

        return one_hot_data

    @staticmethod
    def create_binary(data, threshold):
        """ Thresholds input data"""

        binary_data = (data >= threshold).astype(np.int)

        return binary_data

    @staticmethod
    def load_filenames(directory, template, sort=True):
        """ Load a list of filenames matching the template from a given directory"""

        filenames = os.listdir(directory)
        matched_filenames = [name for name in filenames if name.find(template) > -1]

        if sort:
            matched_filenames = natsorted(matched_filenames)

        return matched_filenames

    @staticmethod
    def fit_dist(data, dist_type='normal'):
        pass



class UCLDatabaseDexnet(object):
    """
    Create custom data-set from dex-net data-set (Mahler, 2017)
    """


    def __init__(self, config, load_from_pkl=False):

        self.grasps = {}
        self.config = config
        self._setup_config()

        # stable poses for every object
        self.stable_poses = {}

        # initialise database object
        self.hdf5_db = db.Hdf5Database(os.path.join(self.database_dir, self.database_filename))

        dataset_name = self.hdf5_db.datasets[0].dataset_name_
        hdf5_group = self.hdf5_db.data_['datasets'][dataset_name]

        # initialise data-set object
        self.hdf5_ds = db.Hdf5Dataset(dataset_name, hdf5_group, cache_dir=self.dataset_cache_dir)

        self.gripper = RobotGripper.load(self.gripper_name, gripper_dir=self.config['grippers_dir'])


    def _setup_config(self):
        """ Read config file and setup class variables """

        self.database_dir = self.config['database_dir']
        self.database_filename = self.config['database_filename']
        self.dataset_cache_dir = self.config['dataset_cache_dir']
        self.gripper_name = self.config['gripper_name']
        self.grasp_metric = self.config['grasp_metric']
        self.cache_datapoints_limit = self.config['cache_datapoints_limit']

        # params related to collision checking
        self._setup_collision_checker_params()

        # camera params
        self.camera_params = self.config['camera_params']
        self.num_image_samples = self.config['num_image_samples']
        self.output_img_height = self.config['output_image_params']['output_img_height']
        self.output_img_width = self.config['output_image_params']['output_img_width']
        self.output_img_crop_width = self.config['output_image_params']['output_img_crop_width']
        self.output_img_crop_height = self.config['output_image_params']['output_img_crop_height']


    def _setup_collision_checker_params(self):
        """ Setup the discrete space over which to sample the collision checking """

        self.approach_steps = []
        self.approach_dist = self.config['approach_dist']
        self.delta_approach = self.config['delta_approach']
        self.max_approach_angle_z = np.deg2rad(self.config['max_approach_angle_z'])
        self.max_approach_angle_y = np.deg2rad(self.config['max_approach_angle_y'])
        self.min_approach_angle_y = -self.max_approach_angle_y
        self.table_mesh = ObjFile(self.config['table_mesh_filename']).read()

        num_samples = self.config['num_approach_samples']


        # get approach angle increments
        if self.max_approach_angle_y == self.min_approach_angle_y:
            approach_inc = 1
        elif num_samples == 1:
            approach_inc = self.max_approach_angle_y - self.min_approach_angle_y + 1
        else:
            approach_inc = (self.max_approach_angle_y - self.min_approach_angle_y) / (num_samples - 1)

        approach_angle = self.min_approach_angle_y

        # create list of approach angles to try for collision checking
        while approach_angle <= self.max_approach_angle_y:
            self.approach_steps.append(approach_angle)
            approach_angle += approach_inc


    def get_object_keys(self):
        pass


    def _get_stable_poses(self):
        """ Get the stable poses in which the object can be placed on the table"""
        for key in self.hdf5_ds.object_keys:
            self.stable_poses[key] = self.hdf5_ds.stable_poses(key)


    def _get_collision_free_grasps(self, obj, all_grasps):
        """ Filters out grasps that are in collision with the table"""

        num_data_points = 0
        valid_grasps = {}

        # initialise collision checker for unachievable grasps. GraspCollisionChecker does not have a remove object
        # method, therefore need to reinitialise Class for each obj
        collision_checker = GraspCollisionChecker(self.gripper)
        collision_checker.set_graspable_object(obj)

        # load stable poses for current object
        stable_poses = self.hdf5_ds.stable_poses(obj.key)

        for pose in stable_poses:

            # aligned_grasps = []
            valid_grasps[pose.id] = []
            # setup table in collision checker (does stp "mean stable pose"?)
            T_obj_stp = pose.T_obj_table.as_frames('obj', 'stp')
            T_obj_table = obj.mesh.get_T_surface_obj(T_obj_stp, delta=self.config['table_offset']).as_frames('obj',
                                                                                                             'table')
            T_table_obj = T_obj_table.inverse()
            collision_checker.set_table(self.config['table_mesh_filename'], T_table_obj)

            # align all grasps with table normal
            # aligned_grasps = [grasp.perpendicular_table(pose) for grasp in all_grasps]

            # get aligned_grasps along with grasp metrics
            for idx, grasp in enumerate(all_grasps[0]):

                found_grasp = False
                grasp.metric_type = 'robust_ferrari_canny'
                grasp.metric = all_grasps[1][idx]
                grasp = grasp.perpendicular_table(pose)

                # angles relative to table normal (see docstring at the top)
                grasp_axis_angle, grasp_approach_angle, _ = grasp.grasp_angles_from_stp_z(pose)

                if np.abs(grasp_approach_angle) > self.max_approach_angle_z:
                    continue

                # check collision along approach directions
                for angle in self.approach_steps:
                    rotated_test_grasp = grasp.grasp_y_axis_offset(angle)
                    in_collision = collision_checker.collides_along_approach(rotated_test_grasp,
                                                                             self.approach_dist,
                                                                             self.delta_approach)
                    # break if at-least one collision free path is found
                    if not in_collision:
                        found_grasp = True
                        break

                # label as bad grasp if no collision free grasp is found
                if found_grasp:
                    valid_grasps[pose.id].append(grasp)
                else:
                    grasp.metric = 0
                    valid_grasps[pose.id].append(grasp)

                num_data_points += 1

        return valid_grasps, num_data_points


    def get_grasps(self, num_objs=float('inf')):
        """ Get grasps for the given object and gripper. Saves the resulting grasping into cache"""

        total_data_points = 0
        current_data_points = 0
        pickle_file_num = 1

        abs_st_time = time.time()
        st_time = time.time()
        # get stable poses for all objects
        self._get_stable_poses()

        # for obj_key in self.hdf5_ds.object_keys:
        for obj_idx, obj in enumerate(self.hdf5_ds):

            print('Starting grasp calculation for %s' % obj.key)

            # get all grasps for obj
            all_grasps = self.hdf5_ds.sorted_grasps(obj.key, self.config['grasp_metric'], self.gripper_name)

            # get collision free grasps
            self.grasps[obj.key], num_data_points = self._get_collision_free_grasps(obj, all_grasps)

            current_data_points += num_data_points
            total_data_points += num_data_points
            # save grasps and metrics in cache so we don't run out of memory
            if current_data_points > self.cache_datapoints_limit or (obj_idx + 1) == self.hdf5_ds.num_objects:

                # cache filenames
                cache_filename = 'grasp_cache' + str(pickle_file_num) + '.pkl'
                grasp_cache_filename = os.path.join(self.dataset_cache_dir + cache_filename)

                with open(grasp_cache_filename, 'wb') as pkl_file:
                    print('Writing pickle file %s' % cache_filename)
                    print('Time taken for %d data points: %s(s)\n' % (current_data_points, (time.time() - st_time)))
                    pkl.dump(self.grasps, pkl_file)

                    pickle_file_num += 1
                    current_data_points = 0
                    # flush grasps to prevent running out of memory
                    self.grasps.clear()
                    st_time = time.time()

            if obj_idx + 1 == num_objs:
                break

        print('Total number of data-point: %d' % total_data_points)
        print('Total number of  pickle files: %d' % pickle_file_num)
        print('Total time taken: %s(s)' % str(time.time() - abs_st_time))


    def get_rendered_images(self, pickle_file_num, total_data_points, grasps=None):
        """ Get rendered images for for all stable poses with valid grasps. Saves the resulting grasping into cache"""

        if type(grasps) is None:
            grasps = self.grasps
        elif type(grasps) is dict:
            pass
        elif type(grasps) is str:
            # load load external pickle file
            grasps = pkl.load(open(grasps, 'rb'))
        else:
            raise TypeError('Unknown type for argument grasps: Must be of a filename (type "str")')

        if not grasps:
            warnings.warn('W: grasps dictionary is empty!')

        current_data_points = 0
        # total_data_points = 0
        # pickle_file_num = 1

        abs_st_time = time.time()
        st_time = time.time()

        # store rendered images and grasps
        obj_renders = {}

        # only get objects for which a grasp exists
        objs = [self.hdf5_ds[key] for key in grasps.keys()]

        for obj_idx, obj in enumerate(objs):

            stable_poses = self.hdf5_ds.stable_poses(obj.key)

            for pose in stable_poses:

                obj_renders[obj.key] = {pose.id: []}

                # grasps for current object
                obj_grasps = grasps[obj.key][pose.id]

                # object pose wrt table
                T_obj_stp = pose.T_obj_table.as_frames('obj', 'stp')
                T_obj_stp = obj.mesh.get_T_surface_obj(T_obj_stp)

                # sample images from camera model accounting for positional uncertainty
                T_table_obj = RigidTransform(from_frame='table', to_frame='obj')
                scene_objs = {'table': SceneObject(self.table_mesh, T_table_obj)}
                uirv = UniformPlanarWorksurfaceImageRandomVariable(obj.mesh,
                                                                   [RenderMode.DEPTH_SCENE],
                                                                   'camera',
                                                                   self.camera_params,
                                                                   stable_pose=pose,
                                                                   scene_objs=scene_objs)

                # sample multiple images (model randomness)
                samples = uirv.rvs(size=self.num_image_samples)

                for sample in samples:

                    # store all grasps for current sample
                    sample_grasps = []

                    # get image
                    depth_img = sample.renders[RenderMode.DEPTH_SCENE].image

                    # get camera transformation
                    T_stp_camera = sample.camera.object_to_camera_pose
                    camera_intr = sample.camera.camera_intr
                    # get center pixels
                    center_x = depth_img.center[1]
                    center_y = depth_img.center[0]
                    
                    # object in camera frame
                    T_obj_camera = T_stp_camera * T_obj_stp.as_frames('obj', 'stp')

                    corrected_camera_intr = camera_intr
                    # recompute intrinsics if image is being cropped and resized
                    if self.config['output_image_params']['resize']:
                        scale = self.output_img_height/float(self.output_img_crop_height)
                        cropped_camera_intr = camera_intr.crop(self.output_img_crop_height, self.output_img_width,
                                                               center_x, center_y)
                        corrected_camera_intr = cropped_camera_intr.resize(scale)

                        # crop image
                        depth_img = depth_img.crop(self.output_img_crop_height, self.output_img_crop_width)
                        # resize image
                        depth_img = depth_img.resize((self.output_img_height, self.output_img_width))

                    for grasp in obj_grasps:

                        # project gripper into camera (T[stp->cam]T[obj->stp])
                        P_grasp_camera = grasp.project_camera(T_obj_camera, camera_intr)

                        # take the cropping into account for grasp center
                        translation_x = center_x - self.output_img_crop_width/2
                        translation_y = center_y - self.output_img_crop_height/2
                        scaled_grasp_center_x = scale * (P_grasp_camera.center.x - translation_x)
                        scaled_grasp_center_y = scale * (P_grasp_camera.center.y - translation_y)
                        # translated_grasp_center = np.array([scaled_grasp_center_x, scaled_grasp_center_y])

                        # get grasp in image space
                        # grasp_center = Point(translated_grasp_center, frame=corrected_camera_intr.frame)
                        sample_grasps.append([scaled_grasp_center_x, scaled_grasp_center_y, P_grasp_camera.angle, P_grasp_camera.depth])

                        current_data_points += 1
                        total_data_points += 1

                    pose_sample = {'image': depth_img,
                                   'grasps': sample_grasps,
                                   'vis': {'camera_intr': corrected_camera_intr}}
                    obj_renders[obj.key][pose.id].append(pose_sample)

            if current_data_points > self.cache_datapoints_limit or (obj_idx + 1) == len(objs):
                # cache filenames
                cache_filename = 'image_cache' + str(pickle_file_num) + '.pkl'
                image_cache_filename = os.path.join(self.dataset_cache_dir + cache_filename)

                with open(image_cache_filename, 'wb') as pkl_file:
                    print('Writing pickle file %s' % cache_filename)
                    print('Time taken for %d data points: %s(s)\n' % (current_data_points, (time.time() - st_time)))
                    pkl.dump(obj_renders, pkl_file)

                    pickle_file_num += 1
                    current_data_points = 0
                    # flush grasps to prevent running out of memory
                    obj_renders.clear()
                    st_time = time.time()

        print('Total number of data-points: %d' % total_data_points)
        print('Total number of  pickle files: %d' % pickle_file_num)
        print('Total time taken: %s(s)' % str(time.time() - abs_st_time))

        return pickle_file_num, total_data_points

    # TODO: Implement get_data method to control data access and file read/write
    # def get_data(self, data_type, num_objs):
    #     """ Get data and save to cache"""
    #     if data_type == 'grasps':
    #         cache_filename = 'grasp_cache'
    #         method_call = self.get_grasps
    #
    #     elif data_type == 'rendered_images':
    #         cache_filename = 'image_cache'
    #         method_call = self.get_rendered_images
    #
    #     else:
    #         raise StandardError('get_data type %s not supported' % data_type)
    #
    #     total_data_points = 0
    #     current_data_points = 0
    #     pickle_file_num = 1
    #
    #     abs_st_time = time.time()
    #     st_time = time.time()
    #
    #     for obj_idx, obj in enumerate(self.hdf5_ds):
    #
    #         data, num_data_points = method_call()
    #
    #         current_data_points += num_data_points
    #         total_data_points += num_data_points
    #         # save grasps and metrics in cache so we don't run out of memory
    #         if current_data_points > self.cache_datapoints_limit or (obj_idx + 1) == self.hdf5_ds.num_objects:
    #
    #             cache_filpath = os.path.join(self.dataset_cache_dir, cache_filename + str(pickle_file_num) + '.pkl')
    #             with open(cache_filpath, 'wb') as pkl_file:
    #                 print('Writing pickle file %s' % cache_filename)
    #                 print('Time taken for %d data points: %s(s)\n' % (current_data_points, (time.time() - st_time)))
    #                 pkl.dump(self.grasps, pkl_file)
    #
    #                 pickle_file_num += 1
    #                 current_data_points = 0
    #                 # flush grasps to prevent running out of memory
    #                 self.grasps.clear()
    #                 st_time = time.time()
    #
    #         if obj_idx + 1 == num_objs:
    #             break
    #
    #     print('Total number of data-point: %d' % total_data_points)
    #     print('Total number of  pickle files: %d' % pickle_file_num)
    #     print('Total time taken: %s(s)' % str(time.time() - abs_st_time))

    def compile_database(self):
        pass

    def visualise(self, images, proj_grasp_img):

        for image in images:
            vis2d.figure()
            vis2d.imshow(image)
            vis2d.grasp(proj_grasp_img)
            vis2d.show()
        pass


if __name__ == '__main__':

    dexnet_config = YamlConfig('/home/noorvir/catkin_ws/src/grasp_ucl/cfg/generate_ucl_dexnet_dataset.yaml')
    gqcnn_config = YamlConfig('/home/noorvir/catkin_ws/src/grasp_ucl/cfg/generate_ucl_gqcnn_dataset.yaml')

    # UCL_DEXNET
    # ucl_denet_db = UCLDatabaseDexnet(dexnet_config)
    # ucl_denet_db.get_grasps()           # get grasps

    # get images for stable object poses with valid grasps
    # ucl_denet_db.get_rendered_images(ucldb.grasps)

    # all_filenames = os.listdir(dexnet_config['dataset_cache_dir'])
    # grasp_cache_filenames =[name for name in all_filenames if name.find('grasp_cache') > -1]
    # grasp_cache_filenames = natsorted(grasp_cache_filenames)
    #
    # pickle_file_num = 1
    # total_data_points = 0
    # for idx, cache_file in enumerate(grasp_cache_filenames):
    #     print('Getting grasp file number %d' % (idx + 1))
    #     grasp_cache_file_path = os.path.join(dexnet_config['dataset_cache_dir'], cache_file)
    #     pickle_file_num, total_data_points = ucl_denet_db.get_rendered_images(pickle_file_num, total_data_points, grasp_cache_file_path)

    # compile database - associate rendered images with grasps and metrics


    # UCL_GQCNN
    ucl_gqcnn_db = UCLDatabaseGQCNN(gqcnn_config)
    # ucl_gqcnn_db.get_metric_stats()                 # get statistics about successful grasps
    # ucl_gqcnn_db.create_images('depth_ims_tf_table', 'depth_ims_stf_{}_table'.format(gqcnn_config['output_img_size']))
    ucl_gqcnn_db.create_labels()                    # create one-hot labels for quality function
    # ucl_gqcnn_db.visualise()
    pass
