from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf

import utils.config as config
import utils.utilities as utils
from utils.logger import Logger


class Model:

    """ Class representing an abstract Model.
    """

    def __init__(self,
                 model_name,
                 main_dir,
                 cost_func='rmse',
                 num_epochs=10,
                 batch_size=100,
                 opt='adam',
                 learning_rate=0.001,
                 momentum=0.1,
                 task='regression',
                 save_summary=False,
                 seed=-1,
                 verbose=0):

        """
        :param model_name: Name of the model, used as filename.
        :param main_dir: Main directory to put the stored_models, data and summary directories.
        :param cost_func:
        :param num_epochs:
        :param batch_size:
        :param opt:
        :param learning_rate:
        :param momentum:
        :param task: 'classification' or 'regression'
        :param save_summary:
        :param seed: positive integer for seeding random generators. Ignored if < 0.
        :param verbose: Level of verbosity. 0 - silent, 1 - print.
        """

        # Validations
        assert model_name is not ''
        assert main_dir is not ''
        assert num_epochs > 0
        assert batch_size > 0
        assert opt in utils.valid_optimization_functions
        assert learning_rate > 0 if opt is not 'momentum' else True
        assert momentum > 0 if opt == 'momentum' else True

        self.model_name = model_name
        self.main_dir = main_dir

        # Model input data
        self._input = None

        # Model output layer
        self._model_output = None

        # Cost function
        self.cost_func = cost_func
        self.cost = None

        # Training parameters
        self.num_epochs = num_epochs
        self.batch_size = batch_size

        # Optimization function
        self.opt_func = opt
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.optimizer = None

        # TensorFlow objects
        self.tf_graph = tf.Graph()
        self.tf_session = None
        self.tf_saver = None

        if seed >= 0:
            np.random.seed(seed)
            tf.set_random_seed(seed)

        self.task = task
        self.verbose = verbose

        # Create the logger
        self.logger = Logger(model_name, verbose)

        # Create directories
        self.models_dir = os.path.join(config.models_dir, main_dir)
        utils.create_dir(self.models_dir)

        self.model_path = os.path.join(self.models_dir, self.model_name)

        self.save_summary = save_summary
        if save_summary:
            self.tf_merged_summaries = None
            self.tf_summary_writer = None
            self.tf_summary_dir = os.path.join(config.summary_dir, main_dir)
            utils.create_dir(self.tf_summary_dir)

    def _create_cost_node(self, ref_input, reg_term=None):

        """ Create the cost function node.
        :param ref_input: reference input placeholder node
        :param reg_term: regularization term
        :return: self
        """

        with tf.name_scope("cost"):
            if self.cost_func == 'cross_entropy':
                cost = -tf.reduce_mean(tf.mul(ref_input, tf.log(tf.clip_by_value(self._model_output, 1e-10, float('inf')))) +
                                       tf.mul((1 - ref_input), tf.log(tf.clip_by_value(1 - self._model_output, 1e-10, float('inf')))))

            elif self.cost_func == 'softmax_cross_entropy':
                softmax = tf.nn.softmax(self._model_output)
                cost = -tf.reduce_mean(tf.mul(ref_input, tf.log(softmax)) +
                                       tf.mul((1 - ref_input), tf.log(1 - softmax)))

            elif self.cost_func == 'rmse':
                cost = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(ref_input, self._model_output))))

            self.cost = cost + reg_term if reg_term is not None else cost
            _ = tf.scalar_summary(self.cost_func, self.cost)

    def _create_optimizer_node(self):

        """ Create the training step node of the network.
        :return: self
        """

        with tf.name_scope("train"):
            if self.opt_func == 'gradient_descent':
                self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost)

            elif self.opt_func == 'ada_grad':
                self.optimizer = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.cost)

            elif self.opt_func == 'momentum':
                self.optimizer = tf.train.MomentumOptimizer(self.learning_rate, self.momentum).minimize(self.cost)

            elif self.opt_func == 'adam':
                self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

            elif self.opt_func == 'rms_prop':
                self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.cost)

            else:
                self.optimizer = None

    def _initialize_tf(self, restore_previous_model=False):

        """ Initialize TensorFlow operations: summaries, init operations, saver, summary_writer.
        Restore a previously trained model if the flag restore_previous_model is true.
        :param restore_previous_model:
                    if true, a previous trained model
                    with the same name of this model is restored from disk to continue training.
        """

        self.tf_session.run(tf.initialize_all_variables())

        self.tf_saver = tf.train.Saver()

        if restore_previous_model:
            self.logger.info('Restoring previous model from %s' % self.model_path)
            self.tf_saver.restore(self.tf_session, self.model_path)

        if self.save_summary:
            self.tf_merged_summaries = tf.merge_all_summaries()
            # Retrieve run identifier
            run_id = 0
            for e in os.listdir(self.tf_summary_dir):
                if e[:3] == 'run':
                    r = int(e[3:])
                    if r > run_id:
                        run_id = r

            run_id += 1
            run_dir = os.path.join(self.tf_summary_dir, 'run' + str(run_id))
            self.logger.info('Tensorboard logs dir for this run is %s' % run_dir)
            self.tf_summary_writer = tf.train.SummaryWriter(run_dir, self.tf_session.graph)

    def _run_validation_cost_and_summaries(self, epoch, feed):

        """ Run the summaries and error computation on the validation set.
        :param epoch: Running epoch
        :param feed: TensorFlow feed_dict
        :return: self
        """

        if self.save_summary:
            result = self.tf_session.run([self.tf_merged_summaries, self.cost], feed_dict=feed)
            summary_str = result[0]
            cost = result[1]

            self.tf_summary_writer.add_summary(summary_str, epoch)

        else:
            cost = self.tf_session.run(self.cost, feed_dict=feed)

        self.logger.info("Validation Cost: {}".format(cost))

    def get_model_parameters(self, graph=None):
        pass
