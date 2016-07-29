from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import errno
import os
from os.path import expanduser

import numpy as np
import tensorflow as tf

import utils.config as config
import utils.utilities as utils


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
                 seed=-1,
                 verbose=0,
                 task='regression'):

        """
        :param model_name: name of the model, used as filename. string, default 'dae'
        :param main_dir: main directory to put the stored_models, data and summary directories
        :param cost_func:
        :param opt:
        :param learning_rate:
        :param momentum:
        :param seed: positive integer for seeding random generators. Ignored if < 0.
        :param verbose: Level of verbosity. 0 - silent, 1 - print accuracy.
        :param task: 'classification' or 'regression'
        """

        # Validations
        assert model_name is not ''
        assert main_dir is not ''
        assert num_epochs > 0
        assert batch_size > 0
        assert opt in utils.valid_optimization_functions
        assert learning_rate > 0 if opt is not 'momentum' else True
        assert momentum > 0 if opt == 'momentum' else True

        # Create directories
        home = os.path.join(expanduser("~"), 'dl_data')

        self.model_name = model_name
        self.main_dir = os.path.join(home, main_dir)
        self.models_dir = os.path.join(home, config.models_dir, main_dir)
        self.data_dir = os.path.join(home, config.data_dir, main_dir)
        self.tf_summary_dir = os.path.join(home, config.summary_dir, main_dir)
        self.model_path = os.path.join(self.models_dir, self.model_name)

        print('Creating %s directory to save/restore models' % self.models_dir)
        self._create_dir(self.models_dir)
        print('Creating %s directory to save model generated data' % self.data_dir)
        self._create_dir(self.data_dir)
        print('Creating %s directory to save tensorboard logs' % self.tf_summary_dir)
        self._create_dir(self.tf_summary_dir)

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

        # tensorflow objects
        self.tf_graph = tf.Graph()
        self.tf_session = None
        self.tf_saver = None
        self.tf_merged_summaries = None
        self.tf_summary_writer = None
        self.tf_summary_writer_available = True

        if seed >= 0:
            np.random.seed(seed)
            tf.set_random_seed(seed)

        self.verbose = verbose
        self.task = task


    def _create_dir(self, dirpath):

        """
        :param dirpath: directory to be created
        """

        try:
            os.makedirs(dirpath)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


    def _create_cost_node(self, ref_input):

        """ Create the cost function node.
        :param model_output: model output node
        :param ref_input: reference input placeholder node
        :return: cost function node
        """

        with tf.name_scope("cost"):
            if self.cost_func == 'cross_entropy':
                self.cost = -tf.reduce_mean(tf.mul(ref_input, tf.log(tf.clip_by_value(self._model_output, 1e-10, float('inf')))) +
                                            tf.mul((1 - ref_input), tf.log(tf.clip_by_value(1 - self._model_output, 1e-10, float('inf')))))

            elif self.cost_func == 'softmax_cross_entropy':
                softmax = tf.nn.softmax(self._model_output)
                self.cost = -tf.reduce_mean(tf.mul(ref_input, tf.log(softmax)) +
                                            tf.mul((1 - ref_input), tf.log(1 - softmax)))

            elif self.cost_func == 'rmse':
                self.cost = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(ref_input, self._model_output))))

            _ = tf.scalar_summary(self.cost_func, self.cost)


    def _create_optmizer_node(self):

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


    def _initialize_tf_utilities_and_ops(self, restore_previous_model=False):

        """ Initialize TensorFlow operations: summaries, init operations, saver, summary_writer.
        Restore a previously trained model if the flag restore_previous_model is true.
        :param restore_previous_model:
                    if true, a previous trained model
                    with the same name of this model is restored from disk to continue training.
        """

        self.tf_merged_summaries = tf.merge_all_summaries()
        init_op = tf.initialize_all_variables()
        self.tf_saver = tf.train.Saver()

        self.tf_session.run(init_op)

        if restore_previous_model:
            self.tf_saver.restore(self.tf_session, self.model_path)

        # Retrieve run identifier
        run_id = 0
        for e in os.listdir(self.tf_summary_dir):
            if e[:3] == 'run':
                r = int(e[3:])
                if r > run_id:
                    run_id = r
        run_id += 1
        run_dir = os.path.join(self.tf_summary_dir, 'run' + str(run_id))
        print('Tensorboard logs dir for this run is %s' % (run_dir))

        self.tf_summary_writer = tf.train.SummaryWriter(run_dir, self.tf_session.graph)


    def _run_validation_cost_and_summaries(self, epoch, feed):

        """ Run the summaries and error computation on the validation set.
        :param feed: tensorflow feed_dict
        :param valid_set: validation data
        :return: self
        """

        result = self.tf_session.run([self.tf_merged_summaries, self.cost], feed_dict=feed)
        summary_str = result[0]
        cost = result[1]

        self.tf_summary_writer.add_summary(summary_str, epoch)

        if self.verbose == 1:
            print("Validation Cost: {}".format(cost))


    def get_model_parameters(self, graph=None):
        pass
