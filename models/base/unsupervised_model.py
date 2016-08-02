from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import utils.utilities as utils
from models.base.model import Model


class UnsupervisedModel(Model):

    """ Class representing an abstract Unsupervised Model.
    """

    def __init__(self,
                 model_name,
                 main_dir,
                 cost_func='rmse',
                 num_epochs=10,
                 batch_size=100,
                 opt='adam',
                 learning_rate=0.01,
                 momentum=0.5,
                 seed=-1,
                 verbose=0,
                 task='regression'):

        super().__init__(model_name=model_name,
                         main_dir=main_dir,
                         cost_func=cost_func,
                         num_epochs=num_epochs,
                         batch_size=batch_size,
                         opt=opt,
                         learning_rate=learning_rate,
                         momentum=momentum,
                         seed=seed,
                         verbose=verbose,
                         task=task)

        # Validation
        assert cost_func in utils.valid_unsupervised_cost_functions

        self._encode = None

    def build_model(self, n_input):

        """ Creates the computational graph for the Unsupervised Model.
        :param n_input: Number of features.
        :return: self
        """

        print('Building {} model'.format(self.model_name))

        self._create_placeholders(n_input)
        self._create_layers(n_input)

        self._create_cost_node(self._input)
        self._create_optimizer_node()

        print('Done building {} model'.format(self.model_name))

    def _create_placeholders(self, n_input):

        """ Create the TensorFlow placeholders for the Unsupervised Model.
        :param n_input: number of features of the first layer
        :return: self
        """

        self._input = tf.placeholder('float', [None, n_input], name='x-input')

    def _create_layers(self, n_input):
        pass

    def fit(self, train_set, valid_set, restore_previous_model=False, graph=None):

        """ Fit the model to the data.
        :param train_set: Training data. shape(n_samples, n_features)
        :param valid_set: Validation data. shape(n_samples, n_features)
        :param restore_previous_model: if true, a previous trained model with the same name of this model
            is restored to continue training.
        :param graph: tensorflow graph object
        :return: self
        """

        print('Starting {} unsupervised training...'.format(self.model_name))

        g = graph if graph is not None else self.tf_graph

        with g.as_default():

            self.build_model(train_set.shape[1])

            with tf.Session() as self.tf_session:
                self._initialize_tf(restore_previous_model)
                self._train_model(train_set, valid_set)
                self.tf_saver.save(self.tf_session, self.model_path)

        print('Done {} unsupervised training...'.format(self.model_name))

    def _train_model(self, train_set, valid_set):
        pass

    def transform(self, data, graph=None):

        """ Transform data according to the model.
        :param data: Data to transform
        :param graph: tf graph object
        :return: transformed data
        """

        g = graph if graph is not None else self.tf_graph

        with g.as_default():
            with tf.Session() as self.tf_session:
                self.tf_saver.restore(self.tf_session, self.model_path)
                encoded_data = self._encode.eval({self._input: data})

        return encoded_data

    def calc_total_cost(self, data, graph=None):

        """ Compute the total reconstruction cost.
        :param data: Input data
        :param graph: tensorflow graph object
        :return: reconstruction cost
        """

        g = graph if graph is not None else self.tf_graph

        with g.as_default():
            with tf.Session() as self.tf_session:
                self.tf_saver.restore(self.tf_session, self.model_path)
                cost = self.cost.eval({self._input: data})

        return cost
