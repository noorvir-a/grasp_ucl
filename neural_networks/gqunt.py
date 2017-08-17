from nn_util import NeuralNet, NeuralNetLayers
from autolab_core import YamlConfig
from datetime import datetime
import tensorflow as tf
import numpy as np
import logging
import signal
import time
import os

# Display logging info
logging.getLogger().setLevel(logging.INFO)


class GQUNtLayers(object):
    """ Helper struct to store layers"""
    def __init__(self):
        pass


class GQUNt(object):
    """ Class to wrap the functionality of the Grasp Uncertainty Alex Net GUAN architecture"""

    def __init__(self, config):

        # initialise network
        self.config = config
        self.architecture = self.config['architecture']
        self.normalization_radius = config['normalisation_radius']
        self.normalization_alpha = config['normalisation_alpha']
        self.normalization_beta = config['normalisation_beta']
        self.normalization_bias = config['normalisation_bias']


        self.nn = NeuralNet(self)
        # get utils for creating layers of net
        self.util = NeuralNetLayers()
        self._graph = self.nn.graph

        # keep track of whether TensorFlow train/test ops have been initialised
        self._queues_initialised = False
        self._pred_network_initialised = False


    def _create_network(self, img_data, pose_data):
        """ Create GUAN-t on top of AlexNet"""

        initialiser = tf.contrib.layers.xavier_initializer()
        # {"conv1_1": {"filt_dim": 7, "num_filt": 64, "pool_size": 1, "pool_stride": 1, "norm": 0},
        #  "conv1_2": {"filt_dim": 5, "num_filt": 64, "pool_size": 2, "pool_stride": 2, "norm": 1},
        #  "conv2_1": {"filt_dim": 3, "num_filt": 64, "pool_size": 1, "pool_stride": 1, "norm": 0},
        #  "conv2_2": {"filt_dim": 3, "num_filt": 64, "pool_size": 1, "pool_stride": 1, "norm": 1}, "pc1": {"out_size": 16},
        #  "pc2": {"out_size": 0}, "fc3": {"out_size": 1024}, "fc4": {"out_size": 1024}, "fc5": {"out_size": 2}}

        self.layers = GQUNtLayers()

        # 1. 1st convolution layer part 1 - conv_1_1
        self.layers.conv1_1 = self.util.convolution(img_data, filter_dim=7, num_filters=64, pool_stride=1, name='conv1_1',
                                                    initialiser=initialiser)

        if self.architecture['conv1_1']['norm']:
            self.layers.conv1_1 = self.util.lrn(self.layers.conv1_1, self.normalization_radius, self.normalization_alpha,
                                                self.normalization_beta, self.normalization_bias, name='lrn1_1')

        # 2. 1st convolution layer part 2 - conv_1_2
        self.layers.conv1_2 = self.util.convolution(self.layers.conv1_1, filter_dim=5, num_filters=64, pool_stride=1, name='conv1_2',
                                                    initialiser=initialiser)

        if self.architecture['conv1_2']['norm']:
            self.layers.conv1_2 = self.util.lrn(self.layers.conv1_2, self.normalization_radius, self.normalization_alpha,
                                                self.normalization_beta, self.normalization_bias, name='lrn1_2')

        # 3. pooling layer
        self.layers.pool1_2 = self.util.max_pool(self.layers.conv1_2, 2, 1, name='pool1_2')

        # 4. 2nd convolution layer part 1 - conv_2_1
        self.layers.conv2_1 = self.util.convolution(self.layers.pool1_2, filter_dim=3, num_filters=64, pool_stride=1, name='conv2_1',
                                                    initialiser=initialiser)

        if self.architecture['conv2_1']['norm']:
            self.layers.conv2_1 = self.util.lrn(self.layers.conv2_1, self.normalization_radius, self.normalization_alpha,
                                                self.normalization_beta, self.normalization_bias, name='lrn2_1')

        # 5. 2nd convolution layer part 2 - conv_2_2
        self.layers.conv2_2 = self.util.convolution(self.layers.conv2_1, filter_dim=3, num_filters=64, pool_stride=1, name='conv2_2',
                                                    initialiser=initialiser)

        if self.architecture['conv2_2']['norm']:
            self.layers.conv2_2 = self.util.lrn(self.layers.conv2_2, self.normalization_radius, self.normalization_alpha,
                                                self.normalization_beta, self.normalization_bias, name='lrn2_2')

        # self.layers.conv2_2_flat = tf.reshape(self.layers.conv2_2, [-1, ])
        self.layers.conv2_2_flat = tf.contrib.layers.flatten(self.layers.conv2_2)

        # 6. 1st fully connected layer for image branch of network - fc3
        self.layers.fc3 = self.util.fully_connected(self.layers.conv2_2_flat, 0, 1024, initialiser, name='fc3')
        self.layers.fc3 = self.util.dropout(self.layers.fc3, self.architecture['fc3']['dropout_rate'])

        # 7. fully connected layer for pose branch - pc1
        self.layers.pc1 = self.util.fully_connected(pose_data, 1, 16, initialiser, name='pc1')

        # 8. 2nd fully connected layer combining image and pose branches - fc4
        img_fc = tf.get_variable('fc4W_im', [1024, 1024], initializer=initialiser)
        pose_fc = tf.get_variable('fc4W_pose', [16, 1024], initializer=initialiser)
        fc4_bias = tf.get_variable('fc4b', [1024], initializer=initialiser)

        self.layers.fc4 = tf.nn.relu(tf.matmul(self.layers.fc3, img_fc) + tf.matmul(self.layers.pc1, pose_fc) + fc4_bias)
        self.layers.fc4 = self.util.dropout(self.layers.fc4, self.architecture['fc4']['dropout_rate'])

        # 9. 3rd fully connected layer - fc5
        self.layers.fc5 = self.util.fully_connected(self.layers.fc4, 1024, self.nn.num_classes, initialiser, name='pc5')

        return self.layers.fc5


    def _get_predition_network(self, reuse=True):
        """ Create network to use for prediction. Uses the same graph but a different method of inputing data"""

        with self._graph.as_default():

            # create prediction graph
            self._pred_input_node = tf.placeholder(tf.float32, [self.nn.batch_size, self.nn.img_width, self.nn.img_height, self.nn.img_channels],
                                                   name='prediction_input_node')
            self._pred_pose_node = tf.placeholder(tf.float32, [self.nn.batch_size, self.nn.pose_dim], name='prediction_pose_node')
            self._pred_label_node = tf.placeholder(tf.float32, [self.nn.batch_size, self.nn.num_classes], name='ground_truth_labels')

            # with tf.variable_scope('training_network'):
            with tf.name_scope('prediction_network'):
                with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
                    self._pred_network_output = self._create_network(self._pred_input_node, self._pred_pose_node)

            # metric operations
            with tf.name_scope('prediction_operations'):

                prediction_outcome = tf.equal(tf.argmax(self._pred_network_output, axis=1),
                                              tf.argmax(self._pred_label_node, axis=1), name='prediction_outcome')
                # predicted labels
                self.pred_predicted_labels = tf.argmax(self._pred_network_output, axis=1)
                # accuracy
                self.pred_accuracy_op = tf.reduce_mean(tf.cast(prediction_outcome, tf.float32), name='accuracy_op')
                # error
                self.pred_error_rate_op = tf.subtract(1.0, self.pred_accuracy_op)

        self._pred_network_initialised = True


    def optimise(self, weights_init='pre_trained'):
        """ Initialise training routine and optimise"""

        # create handler for CNTRL+C signal
        signal.signal(signal.SIGINT, self.nn.signal_handler)

        # setup common filename and logging
        self.nn.model_timestamp = '{:%y-%m-%d-%H:%M:%S}'.format(datetime.now())
        model_dir_name = self.nn.model_timestamp  # directory for current model

        self._model_dir = os.path.join(self.nn.checkpoint_dir, model_dir_name)
        self._log_dir = os.path.join(self._model_dir, 'logs')

        os.mkdir(self._model_dir)
        os.mkdir(self._log_dir)
        file_handler = logging.FileHandler(os.path.join(self._log_dir, 'training_log.log'))
        logging.getLogger().addHandler(file_handler)

        # initialise TensorFlow variables
        self.nn.init_queues()
        self._queues_initialised = True

        # initialise Tensorflow Session and variables
        self.sess = self.nn.open_session()

        self.nn.get_data_metrics()

        self.network_output = self._create_network(self.nn.train_input_node, self.nn.train_pose_node)

        # init validation network
        self._get_predition_network()

        if self.nn.weights_init_type == 'gaussian':
            pass
        elif self.nn.weights_init_type == 'truncated_normal':
            pass
        elif self.nn.weights_init_type == 'xavier':
            pass

        # metrics
        self.nn.init_metric_ops()

        # TODO: find a bet way of initialising these
        self.nn.lr_decay_step = 100000
        batch_num = tf.Variable(0)
        # setup learning rate decay
        if self.nn.exponential_decay:
            self.nn.learning_rate = tf.train.exponential_decay(self.nn.learning_rate, tf.multiply(batch_num, self.nn.batch_size),
                                                               self.nn.lr_decay_step, self.nn.lr_decay_rate,
                                                               name='lr_exponential_decay')
        else:
            # for consistency
            self.nn.learning_rate = tf.constant(self.nn.learning_rate)

        var_list = [var for var in tf.trainable_variables() if var.name.split('/')[0] in self.nn.train_layers]

        # create loss
        with tf.name_scope('loss'):
            self.loss = self.nn.create_loss()
        # create requlariser to penalise large weights

        # create optimiser
        with tf.name_scope('optimiser'):
            optimiser = self.nn.create_optimiser(batch_num, var_list)

        # setup saver
        self.saver = tf.train.Saver()

        self.nn.init_summaries()
        self.nn.launch_tensorboard()

        # setup weight decay

        # number of batches per epoch
        batches_per_epoch = int(self.nn.num_training_samples / self.nn.batch_size)

        # use multiple threads to load data
        coord = tf.train.Coordinator()

        if not self.nn.debug:
            # training-data threads
            qr_train_data = tf.train.QueueRunner(self.nn.train_data_queue,
                                                 [self.nn.train_data_enqueue_op] * self.nn.num_train_data_enqueue_threads)
            # add QueueRunners to default collection
            tf.train.add_queue_runner(qr_train_data)

        # validation-data threads
        qr_val_data = tf.train.QueueRunner(self.nn.val_data_queue,
                                           [self.nn.val_data_enqueue_op] * self.nn.num_val_data_enqueue_threads)
        tf.train.add_queue_runner(qr_val_data)

        # init variables
        self.sess.run(tf.global_variables_initializer())

        # add graph to summary
        self.nn.summariser.add_graph(self.sess.graph)

        # initialise weights (N.B. pretrained weights must be loaded after calling tf.global_variables_initializer()
        if weights_init == 'pre_trained':
            self.nn.load_pretrained_weights()
        elif weights_init == 'checkpoint':
            self.nn.load_weights_from_checkpoint()

        # freeze graph and start threads
        self._graph.finalize()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)

        logging.info('\nWaiting 60 seconds to load queues')
        time.sleep(60)

        # log info about training
        logging.info('------------------------------------------------')
        logging.info('Number of Classes: %s' % str(self.nn.num_classes))
        logging.info('Pose used: %s' % str(bool(int(self.nn.use_pose))))
        logging.info('Number of Training Data-points: %s' % str(self.nn.loader.num_train))
        logging.info('Loss: %s' % self.nn.config['loss'])
        logging.info('Optimiser: %s' % self.nn.config['optimiser'])
        logging.info('Pre-trained layers: %s' % str(self.nn.train_layers))
        logging.info('Dataset Name: %s' % str(self.nn.dataset_name))
        logging.info('Fraction of Dataset Used: %s' % str(self.nn.config['data_used_fraction']))
        logging.info('Fraction of Positive samples: %s' % str(self.nn.config['pos_train_frac']))
        logging.info('Batch Size: %s' % str(self.nn.batch_size))
        logging.info('Learning Rate: %s' % str(self.sess.run(self.nn.learning_rate)))
        logging.info('Learning Rate Exponential Decay: %s' % str(bool(int(self.nn.exponential_decay))))
        logging.info('Momentum Rate: %s' % str(self.nn.momentum_rate))
        logging.info('Weights Initialisation Type: %s' % weights_init)
        logging.info('Debug: %s' % str(bool(int(self.nn.debug))))
        logging.info('Variables to be trained: %s' % str([var.name for var in var_list]))
        logging.info('------------------------------------------------')

        logging.info('\nStarting Optimisation\n')

        # total training steps
        step = 0
        # print trainable variables
        logging.info('Variables to be trained: %s' % str([var.name.split(':')[0] for var in tf.trainable_variables()]))

        st = time.time()
        with tf.device('/gpu:0'):
            # iterate over training epochs
            for epoch in xrange(1, self.nn.config['num_epochs'] + 1):
                # iterate over all batches
                for batch in xrange(1, batches_per_epoch + 1):

                    # ---------------------------------
                    # 1. optimise
                    # ---------------------------------
                    if batch % self.nn.log_frequency != 0 and batch != 1:
                        # only run optimiser for max speed
                        self.sess.run(optimiser)

                    else:
                        run_vars = [optimiser, self.loss, self.nn.accuracy_op, self.nn.merged_train_summaries]

                        _, loss, self.nn.train_accuracy, training_summaries = self.sess.run(run_vars)

                        logging.info(
                            self.nn.get_date_time() + ': epoch = %d, batch = %d, accuracy = %.3f, loss = %.3f, time = %.5f'
                            % (epoch, batch, self.nn.train_accuracy, loss, time.time() - st))

                        # log summaries
                        self.nn.summariser.add_summary(training_summaries, step)
                        st = time.time()

                    # ---------------------------------
                    # 2. validate
                    # ---------------------------------
                    if batch % self.nn.val_frequency == 0:
                        logging.info('------------------------------------------------')
                        logging.info(self.nn.get_date_time() + ': Validating Network ... ')

                        # get data
                        input_batch, pose_batch, label_batch = self.sess.run([self.nn.val_input_node, self.nn.val_pose_node, self.nn.val_label_node])
                        val_accuracy, val_error, _ = self.predict(input_batch, pose_batch, label_batch)

                        logging.info(self.nn.get_date_time() + ': epoch = %d, batch = %d, validation accuracy = %.3f' % (epoch, batch, val_accuracy))
                        logging.info('------------------------------------------------')

                        # log summaries
                        summary = self.sess.run(self.nn.merged_val_summaries, feed_dict={self._pred_input_node: input_batch,
                                                                                         self._pred_pose_node: pose_batch,
                                                                                         self._pred_label_node: label_batch})
                        self.nn.summariser.add_summary(summary, step)

                    # save
                    if batch % self.nn.save_frequency == 0:
                        checkpoint_path = os.path.join(self.nn.checkpoint_dir, model_dir_name, 'model%d.ckpt' % (step + 1))
                        logging.info('------------------------------------------------')
                        logging.info(self.nn.get_date_time() + ': epoch = %d, batch = %d, accuracy = %.3f' % (epoch, batch, self.nn.train_accuracy))
                        logging.info(self.nn.get_date_time() + ': Saving model as %s' % checkpoint_path)
                        logging.info('------------------------------------------------')

                        latest_checkpoint_path = os.path.join(self.nn.checkpoint_dir, model_dir_name,
                                                              'model.ckpt')  # save a copy of the latest model
                        self.saver.save(self.sess, checkpoint_path)
                        self.saver.save(self.sess, latest_checkpoint_path)

                    step += 1

            coord.request_stop()
            coord.join(threads)


    def predict(self, input_batch, pose_label, label_batch, close_sess=False, is_test=False):
        """ Predict """

        with self._graph.as_default():

            if self.sess is None:
                self.nn.open_session()

            # initialise prediction network
            if not self._pred_network_initialised:
                self._get_predition_network(reuse=False)
                self._pred_network_initialised = True

            # load model if testing
            if is_test:
                self.nn.load_weights_from_checkpoint()
                # saver = tf.train.Saver()
                # saver.restore(self.sess, model_path)

            # variables to run
            run_vars = [self.pred_accuracy_op, self.pred_error_rate_op, self._pred_network_output, self.pred_predicted_labels]
            feed_dict = {self._pred_input_node: input_batch, self._pred_pose_node: pose_label, self._pred_label_node: label_batch}
            # run
            run_op_outupt = self.sess.run(run_vars, feed_dict=feed_dict)
            # outputs of run op
            accuracy, error, output, predicted_labels = run_op_outupt

            if close_sess:
                self.sess.close()
                self.sess = None

        return accuracy, error, predicted_labels


