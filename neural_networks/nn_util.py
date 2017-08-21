from grasp_ucl.utils.pre_process import DataLoader
from autolab_core import YamlConfig
from datetime import datetime
import tensorflow as tf
import numpy as np
import pickle as pkl
import logging
import threading
import pandas
import time
import os

# Display logging info
logging.getLogger().setLevel(logging.INFO)


class NeuralNetLayers(object):
    """ Class containing utilities to create layers of neural-nets"""

    def __init__(self):
        pass


    @staticmethod
    def convolution(x, filter_dim, num_filters, conv_stride, initialiser, name, padding='SAME'):
        """Create a convolution layer.

        Adapted from: https://github.com/ethereon/caffe-tensorflow
        """
        # Get number of input channels
        input_channels = int(x.get_shape()[-1])

        # Create lambda function for the convolution
        convolve = lambda i, k: tf.nn.conv2d(i, k, strides=[1, conv_stride, conv_stride, 1], padding=padding)

        with tf.variable_scope(name) as scope:
            # Create tf variables for the weights and biases of the conv layer
            weights = tf.get_variable(name + 'W', shape=[filter_dim, filter_dim, input_channels, num_filters],
                                      initializer=initialiser)
            biases = tf.get_variable(name + 'b', shape=[num_filters], initializer=initialiser)

        conv = convolve(x, weights)

        # Add biases
        bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())

        # Apply relu function
        relu = tf.nn.relu(bias, name=scope.name)

        return relu


    @staticmethod
    def fully_connected(x, num_in, num_out, initialiser, name, bias=True, relu=True):
        """Create a fully connected layer."""
        with tf.variable_scope(name) as scope:

            # Create tf variables for the weights and biases
            weights = tf.get_variable(name + 'W', shape=[num_in, num_out], initializer=initialiser)
            act = tf.matmul(x, weights)

            # add bias
            with tf.name_scope(scope.name):
                if bias:
                    biases = tf.get_variable(name + 'b', [num_out], initializer=initialiser)
                    act = act + biases
                else:
                    act = weights

        if relu:
            # Apply ReLu non linearity
            relu = tf.nn.relu(act)
            return relu
        else:
            return act


    @staticmethod
    def max_pool(x, filter_dim, pool_stride, name, padding='SAME'):
        """Create a max pooling layer."""
        return tf.nn.max_pool(x, ksize=[1, filter_dim, filter_dim, 1], strides=[1, pool_stride, pool_stride, 1], padding=padding,
                              name=name)


    @staticmethod
    def lrn(x, radius, alpha, beta, bias, name='lrn'):
        """Create a local response normalization layer."""
        return tf.nn.local_response_normalization(x, depth_radius=radius, alpha=alpha, beta=beta, bias=bias, name=name)


    @staticmethod
    def dropout(x, keep_prob):
        """Create a dropout layer."""
        return tf.nn.dropout(x, keep_prob)


class NeuralNet(object):
    """ Class to provide shared functionality to create neural nets"""

    def __init__(self, network):

        self._network = network
        self.name = self._network.config['network_name']
        self.config = self._network.config

        self._setup_config()
        # load the dataset config file
        self.dataset_config = YamlConfig(self.dataset_config)[self.name]

        # initialise network
        self._sess = None
        self.graph = tf.get_default_graph()


    def _setup_config(self):
        """ Read config file and setup class variables """

        self.debug = self.config['debug']

        self.dataset_dir = self.config['dataset_dir']
        self.cache_dir = self.config['cache_dir']
        self.dataset_config = self.config['dataset_config']
        self.summary_dir = self.config['summary_dir']
        self.checkpoint_dir = self.config['checkpoint_dir']
        self.model_dir = self.config['model_dir']
        self.pt_model_filename = self.config['pt_model_filename']
        self.checkpoint_filename = self.config['checkpoint_filename']
        self.dataset_name = self.config['dataset_name']

        # data params
        self.metric_sample_size = self.config['metric_sample_size']
        self.data_metrics_filename = self.config['data_metics_filename']
        self.img_max_val = self.config['img_max_val']
        self.img_min_val = self.config['img_min_val']
        self.datapoints_per_file = self.config['datapoints_per_file']
        self.frac_datapoints_from_file = self.config['frac_datapoints_from_file']

        # queues
        self.train_data_queue_capacity = self.config['train_data_queue_capacity']
        self.train_batch_queue_capacity = self.config['train_batch_queue_capacity']
        self.num_train_data_dequeue = self.config['num_train_data_dequeue']
        self.num_train_data_enqueue_threads = self.config['num_train_data_enqueue_threads']
        self.num_train_batch_enqueue_threads = self.config['num_train_batch_enqueue_threads']

        self.val_data_queue_capacity = self.config['val_data_queue_capacity']
        self.val_batch_queue_capacity = self.config['val_batch_queue_capacity']
        self.num_val_data_dequeue = self.config['num_val_data_dequeue']
        self.num_val_data_enqueue_threads = self.config['num_val_data_enqueue_threads']
        self.num_val_batch_enqueue_threads = self.config['num_val_batch_enqueue_threads']

        # training params
        self.val_frequency = self.config['val_frequency']
        self.log_frequency = self.config['log_frequency']
        self.save_frequency = self.config['save_frequency']
        self.batch_size = self.config['batch_size']
        self.pos_train_frac = self.config['pos_train_frac']
        self.weights_init_type = self.config['weights_init_type']
        self.learning_rate = self.config['learning_rate']
        self.lr_decay_rate = self.config['lr_decay_rate']
        self.momentum_rate = self.config['momentum_rate']
        self.exponential_decay = self.config['exponential_decay']

        # architecture
        self.img_width = self.config['architecture']['img_width']
        self.img_height = self.config['architecture']['img_height']
        self.img_channels = self.config['architecture']['img_channels']
        self.num_classes = self.config['architecture']['num_classes']
        self.pose_dim = self.config['architecture']['pose_dim']
        self.train_layers = self.config['architecture']['train_layers']
        self.load_layers = self.config['architecture']['load_layers']

        # use pose input or not
        self.use_pose = 'fc7p' in self.train_layers
        if self.use_pose:
            self.train_layers.append('pc1')


    def signal_handler(self, sig_num, frame):
        """ Handle CNTRL+C signal and shutdown process"""

        logging.info('CNTRL+C signal received')
        # close TensorBoard
        self._close_tensorboard()
        logging.info('Closing TensorFlow session')
        # close session
        self._network.sess.close()
        logging.info('Forcefully exiting')
        exit()


    def get_data_metrics(self):
        """ Get metrics on training data """

        data_metrics_path = os.path.join(self.cache_dir, self.data_metrics_filename)

        if os.path.exists(data_metrics_path):

            data = pkl.load(open(data_metrics_path, 'r'))

            self.img_mean = data['img_mean']
            self.img_stdev = data['img_stdev']

        else:
            num_sample_files = min(self.metric_sample_size, len(self.loader.img_filenames))
            img_filenames = np.random.choice(self.loader.img_filenames, num_sample_files, replace=False)

            img_sum = 0
            num_imgs = 0
            # compute image mean
            for img_file in img_filenames:
                imgs = np.load(os.path.join(self.dataset_dir, img_file))['arr_0']
                img_sum += np.sum(imgs)
                num_imgs += np.shape(imgs)[0]

            self.img_mean = img_sum/float(num_imgs * self.img_width * self.img_height)

            img_sum = 0
            # compute image standard-deviation
            for img_file in img_filenames:
                imgs = np.load(os.path.join(self.dataset_dir, img_file))['arr_0']
                img_sum += np.sum((imgs - self.img_mean)**2)

            self.img_stdev = np.sqrt(img_sum/float(num_imgs * self.img_width * self.img_height))

            data = {'img_mean': self.img_mean, 'img_stdev': self.img_stdev}
            pkl.dump(data, open(data_metrics_path, 'w'))


    def create_loss(self):
        """ Create Loss"""

        # L2-loss
        if self.config['loss'] == 'l2':
            return tf.nn.l2_loss(tf.subtract(self._network.network_output, self.train_label_node))
        # sparse cross-entropy
        elif self.config['loss'] == 'sparse':
            return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.train_label_node, logits=self._network.network_output))
        # cross-entropy loss
        elif self.config['loss'] == 'xentropy':
            return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.train_label_node, logits=self._network.network_output))
        # weighted cross-entropy loss
        elif self.config['loss'] == 'wxentropy':

            if self.num_classes > 2:
                raise ValueError(' Weighted loss is only implemented for binary classification (for now).')

            # ratio of positive training samples to total samples
            weights_ratio = self.pos_train_frac

            # weight training samples based on the distribution of classes in the training data-set
            class_weights = tf.constant([weights_ratio, 1 - weights_ratio])
            labels = tf.multiply(self.train_label_node, class_weights)

            return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels,
                                                                          logits=self._network.network_output))
        else:
            raise ValueError('Loss "%s" not supported' % self.config['loss'])


    def create_optimiser(self, batch, var_list):
        """ Create the optimiser specified in the config file"""

        if self.config['optimiser'] == 'momentum':
            return tf.train.MomentumOptimizer(self.learning_rate, self.momentum_rate).minimize(self._network.loss, global_step=batch,
                                                                                               var_list=var_list)
        elif self.config['optimiser'] == 'adam':
            return tf.train.AdamOptimizer(self.learning_rate).minimize(self._network.loss, global_step=batch, var_list=var_list)

        elif self.config['optimiser'] == 'rmsprop':
            return tf.train.RMSPropOptimizer(self.learning_rate).minimize(self._network.loss, global_step=batch, var_list=var_list)

        elif self.config['optimiser'] == 'sgd':
            return tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self._network.loss, global_step=batch, var_list=var_list)

        else:
            raise ValueError('Optimiser %s not supported' % (self.config['optimiser']))


    def load_pretrained_weights(self):
        """
        Load pretrained weights from file.

        """
        model_path = os.path.join(self.model_dir, self.pt_model_filename)
        logging.info('Loading weights from pre-trained model')
        logging.info('Model path %s: ' % model_path)

        weights = [var.name.split('/')[0] for var in tf.trainable_variables() if var.name.split('/')[0] in self.load_layers]
        weights = list(set(weights))

        # checkpoint reader
        reader = tf.train.NewCheckpointReader(model_path)

        for wt_name in weights:
            with tf.variable_scope(wt_name, reuse=True):

                # special treatment for fc4
                if wt_name != 'fc4':
                    data = reader.get_tensor(wt_name + 'W')
                    var = tf.get_variable(wt_name + 'W')
                    self._network.sess.run(var.assign(data))

                else:
                    data = reader.get_tensor('fc4W_im')
                    var = tf.get_variable('fc4W_im')
                    self._network.sess.run(var.assign(data))

                    data = reader.get_tensor('fc4W_pose')
                    var = tf.get_variable('fc4W_pose')
                    self._network.sess.run(var.assign(data))

                data = reader.get_tensor(wt_name + 'b')
                var = tf.get_variable(wt_name + 'b')
                self._network.sess.run(var.assign(data))

        logging.info('Done.')


    def load_weights_from_checkpoint(self):
        """ Load weights from checkpoint file."""

        checkpoint_path = os.path.join(self.checkpoint_dir, self.checkpoint_filename)
        logging.info('Loading weights from checkpoint')
        logging.info('Checkpoint path %s: ' % checkpoint_path)

        reader = tf.train.NewCheckpointReader(checkpoint_path)

        # var_list = [var for var in tf.trainable_variables() if var.name.split('/')[0] in self.train_layers]

        for op_name in self.load_layers:
            with tf.variable_scope(op_name, reuse=True):

                try:
                    data = reader.get_tensor(op_name + '/biases')
                    var = tf.get_variable('biases')
                    self._network.sess.run(var.assign(data))
                except tf.errors.NotFoundError:
                    logging.warn('Skipping initialising weight: %s' % (op_name + '/biases'))

                data = reader.get_tensor(op_name + '/weights')
                var = tf.get_variable('weights')
                self._network.sess.run(var.assign(data))

        # self.saver.restore(self.sess, checkpoint_path)
        logging.info('Done.')


    def init_summaries(self):
        """ Set-up summaries"""

        # loss
        tf.summary.scalar(self.config['loss'] + '_loss', self._network.loss, collections=['training_summary'])

        # gradients
        gradient_list = self.train_layers
        var_list = [var for var in tf.trainable_variables() if var.name.split('/')[0] in gradient_list]
        gradients = tf.gradients(self._network.loss, var_list)
        gradients = list(zip(gradients, var_list))

        for gradient, var in gradients:
            tf.summary.histogram(var.name[:-2] + '/gradient', gradient, collections=['training_summary'])

        # learning rate
        tf.summary.scalar('learning_rate', self.learning_rate, collections=['training_summary'])
        # accuracy
        tf.summary.scalar('train_accuracy', self.accuracy_op, collections=['training_summary'])
        tf.summary.scalar('val_accuracy', self._network.pred_accuracy_op, collections=['validation_summary'])
        # error
        tf.summary.scalar('train_error', self.error_rate_op, collections=['training_summary'])
        tf.summary.scalar('val_error', self._network.pred_error_rate_op, collections=['validation_summary'])
        # predicted labels
        # tf.summary.text('predicted_labels', self._pred_predicted_labels, collections=['validation_summary'])
        # tf.summary.scalar('ground_truth_labels', self._pred_label_node, collections=['validation_summary'])


        self.merged_train_summaries = tf.summary.merge_all('training_summary')
        self.merged_val_summaries = tf.summary.merge_all('validation_summary')

        # make summary directory
        current_summary_path = os.path.join(self.summary_dir, self._network.model_timestamp)
        os.mkdir(current_summary_path)
        self.summariser = tf.summary.FileWriter(current_summary_path)


    def init_queues(self):
        """ Initialise TensorFlow queues, batches and weights"""

        # setup data loader
        self.loader = DataLoader(self, self.dataset_config)
        self.num_training_samples = self.loader.num_train

        with tf.name_scope('train_data_loader'):
            # wrap python function to load data from numpy files
            train_imgs, train_poses, train_labels = tf.py_func(self.loader.load_train_data, inp=[], Tout=[tf.float32, tf.float32, tf.float32])

        with tf.name_scope('val_data_loader'):
            # wrap python function to load data from numpy files
            val_imgs, val_poses, val_labels = tf.py_func(self.loader.load_val_data, inp=[], Tout=[tf.float32, tf.float32, tf.float32])

        # ---------------------------------
        # 1. Training Queues
        # ---------------------------------
        # queue to load data from file into training buffer
        with tf.name_scope('train_data_queue'):
            if not self.debug:
                self.train_data_queue = tf.FIFOQueue(self.train_data_queue_capacity, dtypes=[tf.float32, tf.float32, tf.float32],
                                                     shapes=[[self.img_width, self.img_height, self.img_channels],
                                                             [self.pose_dim],
                                                             [self.num_classes]], name='train_data_queue')

                self.train_data_enqueue_op = self.train_data_queue.enqueue_many([train_imgs, train_poses, train_labels], name='enqueue_op')
                self.data_queue_size_op = self.train_data_queue.size()

        # queue data into batches from from buffer
        with tf.name_scope('train_batch_queue'):
            if self.debug:
                train_batch_imgs, train_batch_poses, train_batch_labels = tf.py_func(self.loader.debug_load_and_enqueue, inp=[], Tout=[tf.float32, tf.float32, tf.float32])
            else:
                train_batch_imgs, train_batch_poses, train_batch_labels = self.loader.get_train_data()

            self.train_input_node, self.train_pose_node, self.train_label_node = tf.train.batch([train_batch_imgs, train_batch_poses, train_batch_labels],
                                                                                                self.batch_size,
                                                                                                num_threads=self.num_train_batch_enqueue_threads,
                                                                                                capacity=self.train_batch_queue_capacity,
                                                                                                enqueue_many=True,
                                                                                                shapes=[[self.img_width, self.img_height,
                                                                                                         self.img_channels],
                                                                                                        [self.pose_dim],
                                                                                                        [self.num_classes]])

        ## ---------------------------------
        # 2. Validation Queues
        # ---------------------------------
        # queue to load data from file into validation buffer
        with tf.name_scope('val_data_queue'):
            # validation data queue
            self.val_data_queue = tf.FIFOQueue(self.val_data_queue_capacity, dtypes=[tf.float32, tf.float32, tf.float32],
                                               shapes=[[self.img_width, self.img_height, self.img_channels],
                                                       [self.pose_dim],
                                                       [self.num_classes]], name='val_data_queue')

            self.val_data_enqueue_op = self.val_data_queue.enqueue_many([val_imgs, val_poses, val_labels], name='enqueue_op')

        # queue data into batches from from buffer
        with tf.name_scope('val_batch_queue'):
            val_batch_imgs, val_batch_poses, val_batch_labels = self.loader.get_val_data()
            self.val_input_node, self.val_pose_node, self.val_label_node = tf.train.batch([val_batch_imgs, val_batch_poses, val_batch_labels],
                                                                                          self.batch_size,
                                                                                          num_threads=self.num_val_batch_enqueue_threads,
                                                                                          capacity=self.val_batch_queue_capacity,
                                                                                          enqueue_many=True,
                                                                                          shapes=[[self.img_width, self.img_height,
                                                                                                   self.img_channels],
                                                                                                  [self.pose_dim],
                                                                                                  [self.num_classes]])


    def init_metric_ops(self):
        """ Define metrics to assess training, validation and testing """

        # setup accuracy
        with tf.name_scope('accuracy_op'):
            self.prediction_outcome = tf.equal(tf.argmax(self._network.network_output, axis=1), tf.argmax(self.train_label_node, axis=1),
                                               name='prediction_outcome')
            self.accuracy_op = tf.reduce_mean(tf.cast(self.prediction_outcome, tf.float32), name='accuracy_op')

        with tf.name_scope('error_rate'):
            self.error_rate_op = tf.subtract(1.0, self.accuracy_op)


    def open_session(self):
        """ Open TensorFlow Session while accounting for GPU usage if present"""

        # TODO: Implement GPU memory handling
        with self.graph.as_default():
            self._sess = tf.Session()
            self._sess.run(tf.global_variables_initializer())

        return self._sess


    def launch_tensorboard(self):
        """ Launch Tensorboard"""

        logging.info("Launching Tensorboard at localhost:6006")
        os.system('tensorboard --logdir=' + self.summary_dir + " &>/dev/null &")


    @staticmethod
    def _close_tensorboard():
        """ Shut-down Tensorboard """
        logging.info('Closing Tensorboard.')
        tensorboard_id = os.popen('pgrep tensorboard').read()
        os.system('kill ' + tensorboard_id)


    @staticmethod
    def get_date_time():
        return '{:%Y-%m-%d %H:%M:%S}'.format(datetime.now())
