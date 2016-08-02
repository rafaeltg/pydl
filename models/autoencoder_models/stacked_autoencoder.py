from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import utils.utilities as utils
from models.autoencoder_models.autoencoder import Autoencoder
from models.nnet_models.mlp import MLP


class StackedAutoencoder(MLP):

    """ Implementation of Stacked Autoencoders using TensorFlow.
    """

    def __init__(self,
                 model_name='sae',
                 main_dir='sae/',
                 layers=list([128]),
                 enc_act_func=list(['tanh']),
                 dec_act_func=list(['none']),
                 cost_func=list(['rmse']),
                 num_epochs=list([10]),
                 batch_size=list([10]),
                 opt=list(['adam']),
                 learning_rate=list([0.01]),
                 momentum=list([0.5]),
                 rho=list([0.001]),
                 n_beta=list([3.0]),
                 n_lambda=list([0.0001]),
                 hidden_dropout=1.0,
                 finetune_cost_func='rmse',
                 finetune_act_func='relu',
                 finetune_opt='adam',
                 finetune_learning_rate=0.001,
                 finetune_momentum=0.5,
                 finetune_num_epochs=10,
                 finetune_batch_size=100,
                 seed=-1,
                 verbose=0,
                 task='regression'):

        """
        :param layers: list containing the hidden units for each layer
        :param enc_act_func: Activation function for the encoder. ['sigmoid', 'tanh', 'relu', 'none']
        :param dec_act_func: Activation function for the decoder. ['sigmoid', 'tanh', 'none']
        :param cost_func: Cost function. ['cross_entropy', 'rmse', 'softmax_cross_entropy', 'sparse'].
        :param num_epochs: Number of epochs for training.
        :param batch_size: Size of each mini-batch.
        :param opt: Optimizer to use. string, default 'gradient_descent'. ['gradient_descent', 'ada_grad', 'momentum', 'rms_prop']
        :param learning_rate: Initial learning rate.
        :param momentum: 'Momentum parameter.
        :param rho:
        :param n_beta:
        :param n_lambda:
        :param hidden_dropout: hidden layers dropout parameter.
        :param finetune_cost_func: cost function for the fine tunning step. ['cross_entropy', 'rmse', 'softmax_cross_entropy', 'sparse']
        :param finetune_act_func: activation function for the finetuning step. ['sigmoid', 'tanh', 'relu', 'none']
        :param finetune_opt: optimizer for the finetuning phase
        :param finetune_learning_rate: learning rate for the finetuning.
        :param finetune_momentum: momentum for the finetuning.
        :param finetune_num_epochs: Number of epochs for the finetuning.
        :param finetune_batch_size: Size of each mini-batch for the finetuning.
        :param seed: positive integer for seeding random generators. Ignored if < 0.
        :param task: ['regression', 'classification']
        """

        # Finetuning network
        super().__init__(model_name=model_name,
                         main_dir=main_dir,
                         layers=layers,
                         enc_act_func=finetune_act_func,
                         dec_act_func='none',
                         cost_func=finetune_cost_func,
                         num_epochs=finetune_num_epochs,
                         batch_size=finetune_batch_size,
                         opt=finetune_opt,
                         learning_rate=finetune_learning_rate,
                         momentum=finetune_momentum,
                         dropout=hidden_dropout,
                         init_layers=False,
                         verbose=verbose,
                         seed=seed,
                         task=task)

        self.logger.info('{} __init__'.format(__class__.__name__))

        # Autoencoder parameters
        ae_args = {
            'layers':        layers,
            'enc_act_func':  enc_act_func,
            'dec_act_func':  dec_act_func,
            'cost_func':     cost_func,
            'num_epochs':    num_epochs,
            'batch_size':    batch_size,
            'opt':           opt,
            'learning_rate': learning_rate,
            'momentum':      momentum,
            'rho':           rho,
            'n_beta':        n_beta,
            'n_lambda':      n_lambda
        }

        self.ae_args = utils.expand_args(ae_args)

        # Autoencoders list
        self.autoencoders = []
        self.autoencoder_graphs = []

        self.logger.info('Done {} __init__'.format(__class__.__name__))

    def _create_layers(self, n_input, n_output):

        """
        :param n_input:
        :param n_output:
        :return: self
        """

        # Create autoencoder layers
        self._create_autoencoders()

        # Create finetuning network
        super()._create_layers(n_input, n_output)

    def _create_autoencoders(self):

        """ Create Autoencoder Objects used for unsupervised pretraining
        :return: self
        """

        self.logger.info('Creating {} pretrain nodes...'.format(self.model_name))

        self.autoencoders = []
        self.autoencoder_graphs = []

        for l, layer in enumerate(self.ae_args['layers']):

            self.logger.info('l = {}, layer = {}'.format(l, layer))

            self.autoencoders.append(Autoencoder(model_name='{}_ae_{}'.format(self.model_name, l),
                                                 main_dir=self.main_dir,
                                                 n_hidden=layer,
                                                 enc_act_func=self.ae_args['enc_act_func'][l],
                                                 dec_act_func=self.ae_args['dec_act_func'][l],
                                                 cost_func=self.ae_args['cost_func'][l],
                                                 num_epochs=self.ae_args['num_epochs'][l],
                                                 batch_size=self.ae_args['batch_size'][l],
                                                 opt=self.ae_args['opt'][l],
                                                 learning_rate=self.ae_args['learning_rate'][l],
                                                 momentum=self.ae_args['momentum'][l],
                                                 rho=self.ae_args['rho'][l],
                                                 n_beta=self.ae_args['n_beta'][l],
                                                 n_lambda=self.ae_args['n_lambda'][l],
                                                 verbose=self.verbose))

            self.autoencoder_graphs.append(tf.Graph())

        self.logger.info('Done creating {} pretrain nodes...'.format(self.model_name))

    def pretrain(self, train_set, valid_set):

        """ Perform unsupervised pretraining of the stack of denoising autoencoders.
        :param train_set: training set
        :param valid_set: validation set
        :return: return data encoded by the last layer
        """

        self.logger.info('Starting {} unsupervised pretraining...'.format(self.model_name))

        next_train = train_set
        next_valid = valid_set

        for l, autoenc in enumerate(self.autoencoders):

            self.logger.info('Training layer {}'.format(l+1))

            graph = self.autoencoder_graphs[l]

            # Pretrain a single autoencoder
            autoenc.fit(next_train, next_valid, graph=graph)

            with graph.as_default():
                # Get new wieghts and biases
                params = autoenc.get_model_parameters(graph=graph)

                # Set finetuning network paramenters
                self._layer_nodes[l].set_weights(params['enc_w'])
                self._layer_nodes[l].set_biases(params['enc_b'])

                # Encode the data for the next layer.
                next_train = autoenc.transform(data=next_train, graph=graph)
                next_valid = autoenc.transform(data=next_valid, graph=graph)

        self.logger.info('Done {} unsupervised pretraining...'.format(self.model_name))

    def _train_model(self, train_set, train_labels, valid_set, valid_labels):

        """ Train the model.
        :param train_set: training set
        :param train_labels: training labels
        :param valid_set: validation set
        :param valid_labels: validation labels
        :return: self
        """

        self.pretrain(train_set, valid_set)

        super()._train_model(train_set, train_labels, valid_set, valid_labels)
