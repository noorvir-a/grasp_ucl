from alexnet import AlexNet
from grasp_ucl.utils.pre_process import DataLoader
from nn_util import NeuralNets
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


class GUAN(object):
    """ Class to wrap the functionality of the Grasp Uncertainty Alex Net GUAN architecture"""

    def __init__(self, config):

        # initialise network
        self.config = config
        self.nn = NeuralNets(self)
        self._graph = self.nn.graph

        # keep track of whether TensorFlow train/test ops have been initialised
        self._tensorflow_initialised = False
        self._pred_network_initialised = False


    def _create_network(self, input_data):
        """ Create GUAN-t on top of AlexNet"""

        init = tf.contrib.layers.xavier_initializer()
        # initialise raw AlexNet
        alexnet = AlexNet(input_data, self.nn.num_classes, retrain_layers=self.nn.retrain_layers, initialiser=init)

        # network output
        return alexnet.layers['fc8']


    def _get_predition_network(self):
        """ Create network to use for prediction. Uses the same graph but a different method of inputing data"""

        with self._graph.as_default():

            # create prediction graph
            self._pred_input_node = tf.placeholder(tf.float32, [self.nn.batch_size, self.nn.img_width, self.nn.img_height,
                                                                self.nn.img_channels],  name='prediction_input_node')
            self._pred_label_node = tf.placeholder(tf.float32, [self.nn.batch_size, self.nn.num_classes], name='ground_truth_labels')

            # with tf.variable_scope('training_network'):
            with tf.name_scope('prediction_network'):
                with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                    self._pred_network_output = self._create_network(self._pred_input_node)

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

        # setup common filename and logging
        self.nn.model_timestamp = '{:%y-%m-%d-%H:%M:%S}'.format(datetime.now())
        model_dir_name = self.nn.model_timestamp                                        # directory for current model

        self.nn._model_dir = os.path.join(self.nn.checkpoint_dir, model_dir_name)
        self.nn._log_dir = os.path.join(self.nn._model_dir, 'logs')

        os.mkdir(self.nn._model_dir)
        os.mkdir(self.nn._log_dir)
        file_handler = logging.FileHandler(os.path.join(self.nn._log_dir, 'training_log.log'))
        logging.getLogger().addHandler(file_handler)

        # try:

        # initialise TensorFlow variables
        self.nn.init_queues()

        # initialise Tensorflow Session and variables
        self.sess = self.nn.open_session()
        self._tensorflow_initialised = True

        # with tf.variable_scope('training_network'):
        #     tf.get_variable_scope().reuse_variables()
        # self.nn.keep_prob = tf.placeholder(tf.float32, name='drop_out_keep_prob')

        self.nn.get_data_metrics()

        self.nn.network_output = self._create_network(self.nn.train_input_node)

        # init validation network
        self._get_predition_network()

        # metrics
        self.nn.init_metric_ops()

        # TODO: find a bet way of initialising these
        self.nn.lr_decay_step = 100000
        batch_num = tf.Variable(0)
        # setup learning rate decay
        if self.nn.exponential_decay:
            self.nn.learning_rate = tf.train.exponential_decay(self.nn.learning_rate, tf.multiply(batch_num, self.nn.batch_size),
                                                               self.nn.lr_decay_step, self.nn.lr_decay_rate, name='lr_exponential_decay')

        # create loss
        with tf.name_scope('loss'):
            self.nn.loss = self.nn.create_loss()
        # create requlariser to penalise large weights

        # create optimiser
        with tf.name_scope('optimiser'):
            optimiser = self.nn.create_optimiser(batch_num)

        # setup saver
        self.saver = tf.train.Saver()

        self.nn.init_summaries()
        self.nn.launch_tensorboard()

        # initialise weights
        if weights_init == 'pre_trained':
            self.nn.load_pretrained_weights()
        elif weights_init == 'checkpoint':
            self.nn.load_weights_from_checkpoint()
        else:
            if self.nn.weights_init_type == 'gaussian':
                pass
            elif self.nn.weights_init_type == 'truncated_normal':
                pass
            elif self.nn.weights_init_type == 'xavier':
                pass

        # setup weight decay


        # number of batches per epoch
        batches_per_epoch = int(self.nn.num_training_samples/self.nn.batch_size)

        # init variables
        self.sess.run(tf.global_variables_initializer())

        # add graph to summary
        self.nn.summariser.add_graph(self.sess.graph)

        # log info about training
        logging.info('------------------------------------------------')
        logging.info('Number of Classes: %s' % str(self.nn.num_classes))
        logging.info('Number of Training Data-points: %s' % str(self.nn.loader.num_train))
        logging.info('Loss: %s' % self.nn.config['loss'])
        logging.info('Optimiser: %s' % self.nn.config['optimiser'])
        logging.info('Pre-trained layers: %s' % str(self.nn.retrain_layers))
        logging.info('Dataset Name: %s' % str(self.nn.dataset_name))
        logging.info('Fraction of Dataset Used: %s' % str(self.nn.config['data_used_fraction']))
        logging.info('Fraction of Positive samples: %s' % str(self.nn.config['pos_train_frac']))
        logging.info('Batch Size: %s' % str(self.nn.batch_size))
        logging.info('Learning Rate: %s' % str(self.nn.learning_rate))
        logging.info('Learning Rate Exponential Decay: %s' % str(bool(int(self.nn.exponential_decay))))
        logging.info('Momentum Rate: %s' % str(self.nn.momentum_rate))
        logging.info('Weights Initialisation Type: %s' % weights_init)
        logging.info('Debug: %s' % str(bool(int(self.nn.debug))))
        logging.info('------------------------------------------------')

        # use multiple threads to load data
        coord = tf.train.Coordinator()

        if not self.nn.debug:
            # training-data threads
            qr_train_data = tf.train.QueueRunner(self.nn.train_data_queue, [self.nn.train_data_enqueue_op] * self.nn.num_train_data_enqueue_threads)
            # add QueueRunners to default collection
            tf.train.add_queue_runner(qr_train_data)

        # validation-data threads
        qr_val_data = tf.train.QueueRunner(self.nn.val_data_queue, [self.nn.val_data_enqueue_op] * self.nn.num_val_data_enqueue_threads)
        tf.train.add_queue_runner(qr_val_data)

        self.nn.graph.finalize()

        # start threads
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)

        logging.info('Waiting 60 seconds to load queues')
        time.sleep(0)

        logging.info('Starting Optimisation')

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
                    if batch % self.nn.log_frequency != 0:
                        # only run optimiser for max speed
                        self.sess.run(optimiser)

                    else:
                        run_vars = [optimiser, self.nn.loss, self.nn.accuracy_op, self.nn.merged_train_summaries]

                        _, loss, self.nn.train_accuracy, training_summaries = self.sess.run(run_vars)

                        logging.info(self.nn.get_date_time() + ': epoch = %d, batch = %d, accuracy = %.3f, loss = %.3f, time = %.5f'
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
                        input_batch, label_batch = self.sess.run([self.nn.val_input_node, self.nn.val_label_node])
                        val_accuracy, val_error, _ = self.predict(input_batch, label_batch)

                        logging.info(self.nn.get_date_time() + ': epoch = %d, batch = %d, validation accuracy = %.3f' % (epoch, batch, val_accuracy))
                        logging.info('------------------------------------------------')

                        # log summaries
                        summary = self.sess.run(self.nn.merged_val_summaries, feed_dict={self._pred_input_node: input_batch,
                                                                                         self._pred_label_node: label_batch})
                        self.nn.summariser.add_summary(summary, step)

                    # save
                    if batch % self.nn.save_frequency == 0:
                        checkpoint_path = os.path.join(self.nn.checkpoint_dir, model_dir_name, 'model%d.ckpt' % (step + 1))
                        logging.info('------------------------------------------------')
                        logging.info(self.nn.get_date_time() + ': epoch = %d, batch = %d, accuracy = %.3f' % (epoch, batch, self.nn.train_accuracy))
                        logging.info(self.nn.get_date_time() + ': Saving model as %s' % checkpoint_path)
                        logging.info('------------------------------------------------')

                        latest_checkpoint_path = os.path.join(self.nn.checkpoint_dir, model_dir_name, 'model.ckpt')        # save a copy of the latest model
                        self.saver.save(self.sess, checkpoint_path)
                        self.saver.save(self.sess, latest_checkpoint_path)

                    step += 1

            coord.request_stop()
            coord.join(threads)


    def predict(self, input_batch, label_batch, model_path=None):
        """ Predict """

        with self._graph.as_default():

            # init TensorFlow Session
            if not self._tensorflow_initialised:
                self.nn.open_session()

            close_sess = False
            if self.sess is None:
                close_sess = True
                self.nn.open_session()

            # initialise prediction network
            if not self._pred_network_initialised:
                self._get_predition_network()

            # load model
            if model_path is not None:
                saver = tf.train.Saver()
                saver.restore(self.sess, model_path)

            # variables to run
            run_vars = [self.pred_accuracy_op, self.pred_error_rate_op, self._pred_network_output, self.pred_predicted_labels]
            feed_dict = {self._pred_input_node: input_batch, self._pred_label_node: label_batch}
            # run
            run_op_outupt = self.sess.run(run_vars, feed_dict=feed_dict)
            # outputs of run op
            accuracy, error, output, predicted_labels = run_op_outupt

            if close_sess:
                self.sess.close()
                self.nn._sess = None

        return accuracy, error, predicted_labels
