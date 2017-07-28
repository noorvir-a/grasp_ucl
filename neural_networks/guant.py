from alexnet import AlexNet
from grasp_ucl.utils.pre_process import DataLoader
from autolab_core import YamlConfig
from datetime import datetime
import tensorflow as tf
import numpy as np
import logging
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
        self.create_network()
        self._graph = tf.get_default_graph()


    def _setup_config(self):
        """ Read config file and setup class variables """

        self.dataset_dir = self.config['dataset_dir']
        self.cache_dir = self.config['cache_dir']
        self.dataset_config = self.config['dataset_config']
        self.summary_dir = self.config['summary_dir']
        self.checkpoint_dir = self.config['checkpoint_dir']
        self.pt_weights_file = self.config['pt_weights_filename']

        # training params
        self.val_frequency = self.config['val_frequency']
        self.log_frequency = self.config['log_frequency']
        self.save_frequency = self.config['save_frequency']
        self.batch_size = self.config['batch_size']
        self.queue_capacity = self.config['queue_capacity']

        # architecture
        self.img_width = self.config['architecture']['img_width']
        self.img_height = self.config['architecture']['img_height']
        self.img_channels = self.config['architecture']['img_channels']
        self.num_classes = self.config['architecture']['num_classes']
        self.learning_rate = self.config['architecture']['learning_rate']
        self.momentum_rate = self.config['architecture']['momentum_rate']
        self.retrain_layers = self.config['architecture']['retrain_layers']


    def create_network(self):
        """ Create GUAN-t on top of AlexNet"""

        self.input_node = tf.placeholder(tf.float32, [self.batch_size, self.img_width, self.img_height, self.img_channels])
        self.label_node = tf.placeholder(tf.float32, [self.batch_size, self.num_classes])
        self.keep_prob = tf.placeholder(tf.float32)

        # initialise raw AlexNet
        alexnet = AlexNet(self.input_node, self.keep_prob, self.num_classes, retrain_layers=self.retrain_layers)
        # network output
        self.network_output = alexnet.layers['fc8']


    def _create_loss(self):
        """ Create Loss"""

        if self.config['loss'] == 'l2':
            return tf.nn.l2_loss(tf.subtract(self.network_output, self.label_node))
        elif self.config['loss'] == 'sparse':
            return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label_node,
                                                                                 logits=self.network_output))
        elif self.config['loss'] == 'xentropy':
            return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.label_node,
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

            # Check if layer should be trained from scratch
            if op_name not in self.retrain_layers:

                with tf.variable_scope(op_name, reuse=True):

                    # Assign weights/biases to their corresponding tf variable
                    for data in weights_dict[op_name]:

                        # Biases
                        if len(data.shape) == 1:
                            var = tf.get_variable('biases', trainable=False)
                            self.sess.run(var.assign(data))

                        # Weights
                        else:
                            var = tf.get_variable('weights', trainable=False)
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
        tf.summary.scalar(self.config['loss'] + '_loss', self.loss)

        # gradients
        var_list = [var for var in tf.trainable_variables() if var.name.split('/')[0] in self.retrain_layers]
        gradients = tf.gradients(self.loss, var_list)
        gradients = list(zip(gradients, var_list))

        for gradient, var in gradients:
            tf.summary.histogram(var.name[:-2] + '/gradient', gradient)

        # training accuracy
        tf.summary.scalar('train_accuracy', self.train_accuracy)
        # validation accuracy
        tf.summary.scalar('val_accuracy', self.val_accuracy)

        # training error
        # validation error

        self.summariser = tf.summary.FileWriter(self.summary_dir)


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

        # TODO: setup data loading and preprocessing
        self.num_training_samples = self.loader.num_train

        # TODO: use tf.Iterator to get new training batched for now
        # self.input_node = 0
        # self.label_node = 0
        # TODO: implement TensorFlow FIFOQueue to get new batch data
        # with tf.name_scope('data_queue'):
        #     self.queue = tf.FIFOQueue(self.queue_capacity, [tf.float32, tf.float32], shapes=[(self.batch_size,
        #                                                                                  self.img_width,
        #                                                                                  self.img_height,
        #                                                                                  self.img_channels),
        #                                                                                 (self.batch_size,
        #                                                                                  self.num_classes)])
        #     self.enqueue_data = self.queue.enqueue([])
        #     self.input_node, self.label_node = self.queue.dequeue()


        # setup accuracy
        with tf.name_scope('accuracy_op'):
            prediction_outcome = tf.equal(tf.argmax(self.network_output), tf.argmax(self.label_node), name='prediction_outcome')
            self.accuracy_op = tf.reduce_mean(tf.cast(prediction_outcome, tf.float32), name='accuracy_op')

        with tf.name_scope('error_rate'):
            self.error_rate_op = tf.subtract(1.0, self.accuracy_op)

        with tf.name_scope('train_accuracy'):
            self.train_accuracy = tf.placeholder(tf.float32)

        with tf.name_scope('val_accuracy'):
            self.val_accuracy = tf.placeholder(tf.float32)

        # setup saver
        self.saver = tf.train.Saver()

        # initialise Tensorflow Session and variables
        self.sess = self._open_session()


    def _launch_tensorboard(self):
        """ Launch Tensorboard"""

        logging.info("Launching Tensorboard at localhost:6006")
        os.system('tensorboard --logdir=' + self.summary_dir + " &>/dev/null &")


    def optimise(self, weights_init='pre_trained'):
        """ Initialise training routine and optimise"""

        try:
            # create loss
            with tf.name_scope('loss'):
                self.loss = self._create_loss()

            # create requlariser to penalise large weights

            # create optimiser
            with tf.name_scope('optimiser'):
                optimiser = self._create_optimiser()

            # define accuracy

            # initialise TensorFlow variables
            self._init_tensorflow()
            # setup weight decay (optional)

            self._init_summaries()
            self._launch_tensorboard()
            # add graph to summary

            # number of batches per epoch
            batches_per_epoch = int(self.num_training_samples/self.batch_size)


            # initialise weights
            if weights_init == 'pre_trained':
                self._load_pretrained_weights()
            elif weights_init == 'checkpoint':
                self._load_weights_from_checkpoint()
            else:
                self._init_weights()

            # init variables
            self.sess.run(tf.global_variables_initializer())

            # total training steps
            step = 0


            # iterate over training epochs
            for epoch in xrange(self.config['num_epochs']):

                # iterate over all batches
                for batch in xrange(batches_per_epoch):

                    # load next training batch
                    self.input_node, self.label_node = self.loader.get_next_batch()

                    # train
                    _, loss, self.train_accuracy = self.sess.run([optimiser, self.loss, self.accuracy_op], feed_dict={self.input_node,
                                                                                                                      self.label_node})

                    # validate network
                    if batch % self.val_frequency == 0:
                        logging.info('------------------------------------------------')
                        logging.info(self.get_date_time() + ': Validating Network ... ')
                        loss, self.val_accuracy = self.sess.run([self.loss, self.accuracy_op], feed_dict={self.input_node,
                                                                                                          self.label_node})
                        logging.info(self.get_date_time() + ': epoch = %d, batch = %d, accuracy = %.3f' % (epoch, batch, self.train_accuracy))
                        logging.info('------------------------------------------------')

                    if batch % self.log_frequency == 0:
                        logging.info(self.get_date_time() + ': epoch = %d, batch = %d, accuracy = %.3f' % (epoch, batch, self.train_accuracy))

                    if batch % self.save_frequency == 0:
                        checkpoint_path = os.path.join(self.checkpoint_dir, 'model%d.ckpt' % step)
                        logging.info('------------------------------------------------')
                        logging.info(self.get_date_time() + ': epoch = %d, batch = %d, accuracy = %.3f' % (epoch, batch, self.train_accuracy))
                        logging.info(self.get_date_time() + ': Saving model as %s' % checkpoint_path)
                        logging.info('------------------------------------------------')

                        self.saver.save(self.sess, checkpoint_path)
                        os.path.join(self.checkpoint_dir, 'model.ckpt')        # save a copy of the latest model

                    step += 1

        except Exception as err:
            logging.error(str(err))
            # close TensorBoard
            self._close_tensorboard()
            # close TensorFlow Session
            self.sess.close()


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

    guant = GUANt(guant_config)
    guant.optimise(weights_init='pre_trained')
