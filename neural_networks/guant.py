from alexnet import AlexNet
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


class GUANt(object):
    """ Class to wrap the functionality of the Transformed - Grasp Uncertainty Alex Net GUAN-t architecture"""

    def __init__(self, config):

        self.name = 'guant'
        self.config = config
        self._setup_config()

        # load the dataset config file
        self.dataset_config = YamlConfig(self.dataset_config)[self.name]

        # initialise network
        self._sess = None
        # keep track of whether TensorFlow train/test ops have been initialised
        self._tensorflow_initialised = False
        self._pred_network_initialised = False
        self._graph = tf.get_default_graph()


    def _setup_config(self):
        """ Read config file and setup class variables """

        self.debug = self.config['debug']

        self.dataset_dir = self.config['dataset_dir']
        self.cache_dir = self.config['cache_dir']
        self.dataset_config = self.config['dataset_config']
        self.summary_dir = self.config['summary_dir']
        self.checkpoint_dir = self.config['checkpoint_dir']
        self.pt_weights_file = self.config['pt_weights_filename']
        self.dataset_name = self.config['dataset_name']

        # data params
        self.metric_sample_size = self.config['metric_sample_size']
        self.data_metrics_filename = self.config['data_metics_filename']
        self.img_max_val = self.config['img_max_val']
        self.img_min_val = self.config['img_min_val']
        self.datapoints_per_file = self.config['datapoints_per_file']

        # queues
        self.num_data_dequeue = self.config['num_data_dequeue']
        self.data_queue_capacity = self.config['data_queue_capacity']
        self.batch_queue_capacity = self.config['batch_queue_capacity']
        self.num_data_enqueue_threads = self.config['num_data_enqueue_threads']
        self.num_batch_enqueue_threads = self.config['num_batch_enqueue_threads']

        # training params
        self.val_frequency = self.config['val_frequency']
        self.log_frequency = self.config['log_frequency']
        self.save_frequency = self.config['save_frequency']
        self.batch_size = self.config['batch_size']
        self.pos_train_frac = self.config['pos_train_frac']

        # architecture
        self.img_width = self.config['architecture']['img_width']
        self.img_height = self.config['architecture']['img_height']
        self.img_channels = self.config['architecture']['img_channels']
        self.num_classes = self.config['architecture']['num_classes']
        self.learning_rate = self.config['architecture']['learning_rate']
        self.momentum_rate = self.config['architecture']['momentum_rate']
        self.exponential_decay = self.config['architecture']['exponential_decay']
        self.retrain_layers = self.config['architecture']['retrain_layers']
        self.skip_layers = self.config['architecture']['skip_layers']

    def _get_data_metrics(self):
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


    def _create_loss(self):
        """ Create Loss"""

        # L2-loss
        if self.config['loss'] == 'l2':
            return tf.nn.l2_loss(tf.subtract(self.network_output, self.label_node))
        # sparse cross-entropy
        elif self.config['loss'] == 'sparse':
            return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label_node, logits=self.network_output))
        # cross-entropy loss
        elif self.config['loss'] == 'xentropy':
            return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.label_node, logits=self.network_output))
        # weighted cross-entropy loss
        elif self.config['loss'] == 'wxentropy':

            if self.num_classes > 2:
                raise ValueError(' Weighted loss is only implemented for binary classification (for now).')

            # ratio of positive training samples to total samples
            weights_ratio = self.pos_train_frac

            # weight training samples based on the distribution of classes in the training data-set
            class_weights = tf.constant([weights_ratio, 1 - weights_ratio])
            labels = tf.multiply(self.label_node, class_weights)

            return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels,
                                                                          logits=self.network_output))
        else:
            raise ValueError('Loss "%s" not supported' % self.config['loss'])


    def _create_optimiser(self):
        """ Create the optimiser specified in the config file"""

        if self.config['optimiser'] == 'momentum':
            return tf.train.MomentumOptimizer(self.learning_rate, self.momentum_rate).minimize(self.loss)
        elif self.config['optimiser'] == 'adam':
            return tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        elif self.config['optimiser'] == 'rmsprop':
            return tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)
        else:
            raise ValueError('Optimiser %s not supported' % (self.config['optimiser']))


    def _load_pretrained_weights(self):
        """
        Load pretrained weights from file.

        """

        #https: // www.tensorflow.org / programmers_guide / variables  # choosing_which_variables_to_save_and_restore
        #all_vars = tf.all_variables()
        #var_to_restore = [v for v in all_vars if not v.name.startswith('xxx')]
        #saver = tf.train.Saver(var_to_restore)

        # Load the weights into memory
        weights_dict = np.load(self.pt_weights_file, encoding='bytes').item()

        # Loop over all layer names stored in the weights dict
        for op_name in weights_dict:

            if op_name in self.skip_layers:
                continue

            with tf.variable_scope(op_name, reuse=True):

                is_trainable = False
                # Check if weights in this layer should be modified
                if op_name in self.retrain_layers:
                    is_trainable = True

                # Assign weights/biases to their corresponding tf variable
                for data in weights_dict[op_name]:

                    # Biases
                    if len(data.shape) == 1:
                        var = tf.get_variable('biases', trainable=is_trainable)
                        self.sess.run(var.assign(data))

                    # Weights
                    else:
                        var = tf.get_variable('weights', trainable=is_trainable)
                        self.sess.run(var.assign(data))


    def _load_weights_from_checkpoint(self):
        """ Load weights from checkpoint file."""
        pass


    def _init_weights(self):
        """
        Initialise weights as specified in the config file.

        """
        pass

    def _init_summaries(self):
        """ Set-up summaries"""

        # loss
        tf.summary.scalar(self.config['loss'] + '_loss', self.loss, collections=['training_summary'])

        # gradients
        gradient_list = self.retrain_layers + ['fc8']
        var_list = [var for var in tf.trainable_variables() if var.name.split('/')[0] in gradient_list]
        gradients = tf.gradients(self.loss, var_list)
        gradients = list(zip(gradients, var_list))

        for gradient, var in gradients:
            tf.summary.histogram(var.name[:-2] + '/gradient', gradient, collections=['training_summary'])

        # accuracy
        tf.summary.scalar('train_accuracy', self.accuracy_op, collections=['training_summary'])
        tf.summary.scalar('val_accuracy', self._pred_accuracy_op, collections=['validation_summary'])
        # error
        tf.summary.scalar('train_error', self.error_rate_op, collections=['training_summary'])
        tf.summary.scalar('val_error', self._pred_error_rate_op, collections=['validation_summary'])
        # predicted labels
        # tf.summary.text('predicted_labels', self._pred_predicted_labels, collections=['validation_summary'])
        # tf.summary.scalar('ground_truth_labels', self._pred_label_node, collections=['validation_summary'])


        self.merged_train_summaries = tf.summary.merge_all('training_summary')
        self.merged_val_summaries = tf.summary.merge_all('validation_summary')

        # make summary directory
        current_summary_path = os.path.join(self.summary_dir, self.model_timestamp)
        os.mkdir(current_summary_path)
        self.summariser = tf.summary.FileWriter(current_summary_path)


    def _open_session(self):
        """ Open TensorFlow Session while accounting for GPU usage if present"""

        # TODO: Implement GPU memory handling
        with self._graph.as_default():
            self._sess = tf.Session()
            self._sess.run(tf.global_variables_initializer())

        return self._sess


    def _init_tensorflow(self):
        """ Initialise TensorFlow queues, batches and weights"""

        # setup data loader
        self.loader = DataLoader(self, self.dataset_config)
        self.num_training_samples = self.loader.num_train

        # initialise Tensorflow Session and variables
        self.sess = self._open_session()
        self._tensorflow_initialised = True

        # setup queue to load data from file into buffer
        with tf.name_scope('data_queue'):
            self.data_queue = tf.RandomShuffleQueue(self.data_queue_capacity, 0, dtypes=[tf.float32, tf.float32],
                                                    shapes=[[self.img_width, self.img_height, self.img_channels],
                                                            [self.num_classes]])

            # wrap python function to load data from numpy files
            train_imgs, train_labels = tf.py_func(self.loader.load_data, inp=[], Tout=[tf.float32, tf.float32])
            self.data_enqueue_op = self.data_queue.enqueue_many([train_imgs, train_labels], name='enqueue_op')
            self.data_queue_size_op = self.data_queue.size()

        # setup operations to queue data into batches from from data_queue
        with tf.name_scope('batch_queue'):
            self.batch_queue = tf.FIFOQueue(self.batch_queue_capacity, [tf.float32, tf.float32],
                                            shapes=[[self.img_width, self.img_height, self.img_channels], [self.num_classes]])

            imgs, labels = self.loader.get_batch()
            self.batch_enqueue_op = self.batch_queue.enqueue_many([imgs, labels], name='enqueue_op')
            self.input_node, self.label_node = self.batch_queue.dequeue_many(n=self.batch_size, name='dequeue_op')
            self.batch_queue_size_op = self.batch_queue.size()


    def _init_metric_ops(self):
        """ Define metrics to assess training, validation and testing """

        # setup accuracy
        with tf.name_scope('accuracy_op'):
            self.prediction_outcome = tf.equal(tf.argmax(self.network_output, axis=1), tf.argmax(self.label_node, axis=1),
                                               name='prediction_outcome')
            self.accuracy_op = tf.reduce_mean(tf.cast(self.prediction_outcome, tf.float32), name='accuracy_op')

        with tf.name_scope('error_rate'):
            self.error_rate_op = tf.subtract(1.0, self.accuracy_op)


    def _launch_tensorboard(self):
        """ Launch Tensorboard"""

        logging.info("Launching Tensorboard at localhost:6006")
        os.system('tensorboard --logdir=' + self.summary_dir + " &>/dev/null &")


    def _get_predition_network(self):
        """ Create network to use for prediction. Uses the same graph but a different method of inputing data"""

        with self._graph.as_default():

            # create prediction graph
            self._pred_input_node = tf.placeholder(tf.float32, [self.batch_size, self.img_width, self.img_height, self.img_channels],
                                                   name='prediction_input_node')
            self._pred_label_node = tf.placeholder(tf.float32, [self.batch_size, self.num_classes], name='ground_truth_labels')

            # with tf.variable_scope('training_network'):
            with tf.name_scope('prediction_network'):
                with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                    self._pred_network_output = self.create_network(self._pred_input_node)

            # metric operations
            with tf.name_scope('prediction_operations'):

                prediction_outcome = tf.equal(tf.argmax(self._pred_network_output, axis=1),
                                              tf.argmax(self._pred_label_node, axis=1), name='prediction_outcome')
                # predicted labels
                self._pred_predicted_labels = tf.argmax(self._pred_network_output, axis=1)
                # accuracy
                self._pred_accuracy_op = tf.reduce_mean(tf.cast(prediction_outcome, tf.float32), name='accuracy_op')
                # error
                self._pred_error_rate_op = tf.subtract(1.0, self._pred_accuracy_op)

        self._pred_network_initialised = True


    def create_network(self, input_data):
        """ Create GUAN-t on top of AlexNet"""

        init = tf.contrib.layers.xavier_initializer()
        # initialise raw AlexNet
        alexnet = AlexNet(input_data, self.num_classes, retrain_layers=self.retrain_layers, initialiser=init)

        # network output
        return alexnet.layers['fc8']


    def optimise(self, weights_init='pre_trained'):
        """ Initialise training routine and optimise"""

        # setup common filename and logging
        self.model_timestamp = '{:%y-%m-%d-%H:%M:%S}'.format(datetime.now())
        model_dir_name = self.model_timestamp                                        # directory for current model

        self._model_dir = os.path.join(self.checkpoint_dir, model_dir_name)
        self._log_dir = os.path.join(self._model_dir, 'logs')

        os.mkdir(self._model_dir)
        os.mkdir(self._log_dir)
        file_handler = logging.FileHandler(os.path.join(self._log_dir, 'training_log.log'))
        logging.getLogger().addHandler(file_handler)

        # try:

        # initialise TensorFlow variables
        self._init_tensorflow()
        # with tf.variable_scope('training_network'):
        #     tf.get_variable_scope().reuse_variables()
        # self.keep_prob = tf.placeholder(tf.float32, name='drop_out_keep_prob')

        self._get_data_metrics()

        self.network_output = self.create_network(self.input_node)

        # init validation network
        self._get_predition_network()

        # metrics
        self._init_metric_ops()

        # create loss
        with tf.name_scope('loss'):
            self.loss = self._create_loss()
        # create requlariser to penalise large weights

        # create optimiser
        with tf.name_scope('optimiser'):
            optimiser = self._create_optimiser()

        # setup saver
        self.saver = tf.train.Saver()
        # setup weight decay (optional)

        self._init_summaries()
        self._launch_tensorboard()

        # initialise weights
        if weights_init == 'pre_trained':
            self._load_pretrained_weights()
        elif weights_init == 'checkpoint':
            self._load_weights_from_checkpoint()
        elif weights_init == 'gaussian':
            self._init_weights()

        # number of batches per epoch
        batches_per_epoch = int(self.num_training_samples/self.batch_size)

        # init variables
        self.sess.run(tf.global_variables_initializer())

        # add graph to summary
        self.summariser.add_graph(self.sess.graph)

        # log info about training
        logging.info('------------------------------------------------')
        logging.info('Number of Classes: %s' % str(self.num_classes))
        logging.info('Number of Training Data-points: %s' % str(self.loader.num_train))
        logging.info('Loss: %s' % self.config['loss'])
        logging.info('Optimiser: %s' % self.config['optimiser'])
        logging.info('Pre-trained layers: %s' % str(self.retrain_layers))
        logging.info('Dataset Name: %s' % str(self.dataset_name))
        logging.info('Fraction of Dataset Used: %s' % str(self.config['data_used_fraction']))
        logging.info('Fraction of Positive samples: %s' % str(self.config['pos_train_frac']))
        logging.info('Batch Size: %s' % str(self.batch_size))
        logging.info('Learning Rate: %s' % str(self.learning_rate))
        logging.info('Learning Rate Exponential Decay: %s'% str(bool(int(self.exponential_decay))))
        logging.info('Momentum Rate: %s' % str(self.momentum_rate))
        logging.info('Weights Initialisation Type: %s' % weights_init)
        logging.info('Debug: %s' % str(bool(int(self.debug))))
        logging.info('------------------------------------------------')

        # use multiple threads to load data
        coord = tf.train.Coordinator()
        qr_data = tf.train.QueueRunner(self.data_queue, [self.data_enqueue_op] * self.num_data_enqueue_threads)
        qr_batch = tf.train.QueueRunner(self.batch_queue, [self.batch_enqueue_op] * self.num_batch_enqueue_threads)

        # add QueueRunners to default collection
        tf.train.add_queue_runner(qr_data)
        tf.train.add_queue_runner(qr_batch)

        # create and start threads
        self.data_enqueue_threads = qr_data.create_threads(self.sess, coord=coord, start=True)
        self.batch_enqueue_threads = qr_batch.create_threads(self.sess, coord=coord, start=True)

        logging.info('Waiting 60 seconds to load queues')
        time.sleep(60)

        logging.info('Starting Optimisation')

        # total training steps
        step = 0

        # print trainable variables
        logging.info('Variables to be trained: %s' % str([var.name.split(':')[0] for var in tf.trainable_variables()]))

        with tf.device('/gpu:0'):
            # iterate over training epochs
            for epoch in xrange(1, self.config['num_epochs'] + 1):

                # iterate over all batches
                for batch in xrange(1, batches_per_epoch + 1):

                    # ---------------------------------
                    # 1. optimise
                    # ---------------------------------
                    # variables to run
                    run_vars = [optimiser, self.loss, self.accuracy_op, self.network_output, self.prediction_outcome,
                                 self.data_queue_size_op, self.batch_queue_size_op, self.merged_train_summaries]

                    # TODO: dequeue data only once instead of three times
                    run_op_outupt = self.sess.run(run_vars)
                    # outputs of run op
                    _, loss, self.train_accuracy, output, prediction_outcome, data_queue_size, batch_queue_size, training_summaries = run_op_outupt

                    # ---------------------------------
                    # 2. validate
                    # ---------------------------------
                    if batch % self.val_frequency == 0:

                        # get data
                        input_batch, label_batch = self.loader.get_next_val_batch()
                        val_accuracy, val_error, _ = self.predict(input_batch, label_batch)

                        logging.info('------------------------------------------------')
                        logging.info(self.get_date_time() + ': Validating Network ... ')
                        logging.info(self.get_date_time() + ': epoch = %d, batch = %d, validation accuracy = %.3f' % (epoch, batch, val_accuracy))
                        logging.info('------------------------------------------------')

                        # log summaries
                        summary = self.sess.run(self.merged_val_summaries, feed_dict={self._pred_input_node: input_batch,
                                                                                      self._pred_label_node: label_batch})
                        self.summariser.add_summary(summary, step)

                    # log
                    if batch % self.log_frequency == 0:
                        logging.info(self.get_date_time() + ': epoch = %d, batch = %d, accuracy = %.3f, loss = %.3f, data_queue_size = %d, batch_queue_size = %d'
                                     % (epoch, batch, self.train_accuracy, loss, data_queue_size, batch_queue_size))

                        # log summaries
                        self.summariser.add_summary(training_summaries, step)

                    # save
                    if batch % self.save_frequency == 0:
                        checkpoint_path = os.path.join(self.checkpoint_dir, model_dir_name, 'model%d.ckpt' % (step + 1))
                        logging.info('------------------------------------------------')
                        logging.info(self.get_date_time() + ': epoch = %d, batch = %d, accuracy = %.3f' % (epoch, batch, self.train_accuracy))
                        logging.info(self.get_date_time() + ': Saving model as %s' % checkpoint_path)
                        logging.info('------------------------------------------------')

                        latest_checkpoint_path = os.path.join(self.checkpoint_dir, model_dir_name, 'model.ckpt')        # save a copy of the latest model
                        self.saver.save(self.sess, checkpoint_path)
                        self.saver.save(self.sess, latest_checkpoint_path)

                    step += 1

            coord.request_stop()
            coord.join(self.data_enqueue_threads)
            coord.join(self.batch_enqueue_threads)

        # except Exception as err:
        #     logging.error(str(err))
        #     # close TensorBoard
        #     self._close_tensorboard()
        #     # close TensorFlow Session
        #     self.sess.close()


    def predict(self, input_batch, label_batch, model_path=None):
        """ Predict """

        with self._graph.as_default():

            # init TensorFlow Session
            if not self._tensorflow_initialised:
                self._init_tensorflow()

            close_sess = False
            if self._sess is None:
                close_sess = True
                self._open_session()

            # initialise prediction network
            if not self._pred_network_initialised:
                self._get_predition_network()

            # load model
            if model_path is not None:
                saver = tf.train.Saver()
                saver.restore(self.sess, model_path)

            # variables to run
            run_vars = [self._pred_accuracy_op, self._pred_error_rate_op, self._pred_network_output, self._pred_predicted_labels]
            feed_dict = {self._pred_input_node: input_batch, self._pred_label_node: label_batch}
            # run
            run_op_outupt = self.sess.run(run_vars, feed_dict=feed_dict)
            # outputs of run op
            accuracy, error, output, predicted_labels = run_op_outupt

            if close_sess:
                self.sess.close()
                self._sess = None

        return accuracy, error, predicted_labels


    @staticmethod
    def _close_tensorboard():
        """ Shut-down Tensorboard """
        logging.info('Closing Tensorboard.')
        tensorboard_id = os.popen('pgrep tensorboard').read()
        os.system('kill ' + tensorboard_id)


    @staticmethod
    def get_date_time():
        return '{:%Y-%m-%d %H:%M:%S}'.format(datetime.now())



if __name__ == '__main__':

    guant_config = YamlConfig('/home/noorvir/catkin_ws/src/grasp_ucl/cfg/guant.yaml')


    ####################
    # 1. Train
    ####################
    guant = GUANt(guant_config)
    guant.optimise(weights_init='pre_trained')

    ####################
    # 2. Test
    ####################
    # store metrics over multiple trails in lists
    # accuracy_list = []
    # error_list = []
    # gt_labels_list = []
    # predicted_labels_list = []
    #
    # test_data_loader = DataLoader(guant, guant_config['dataset_config'])
    # num_test_trials = 20
    #
    # for trial in xrange(num_test_trials):
    #
    #     test_input_batch, test_label_batch = test_data_loader.get_next_batch()
    #     accuracy, error, predicted_labels = guant.predict(test_input_batch, test_label_batch, '/home/noorvir/tf_models/GUAN-t/pre_trained/')
    #
    #     accuracy_list.append(accuracy)
    #     error_list.append(error)
    #     gt_labels_list.append(test_label_batch)
    #     predicted_labels_list.append(predicted_labels)
    #
    # label_batch_pd = pandas.Series(gt_labels_list, name='Actual')
    # predicted_labels_pd = pandas.Series(predicted_labels_list, name='Predicted')
    #
    # confusion_mat = pandas.crosstab(label_batch_pd, predicted_labels_pd, rownames=['Actual'], colnames=['Predicted'],  margins=True)

