from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Dense, Dropout
from keras.regularizers import l1l2

import pydl.utils.utilities as utils
from pydl.models.autoencoder_models.autoencoder import Autoencoder
from pydl.models.base.supervised_model import SupervisedModel


class StackedAutoencoder(SupervisedModel):

    """ Implementation of Stacked Autoencoders.
    """

    def __init__(self,
                 name='sae',
                 layers=list([64, 32]),
                 ae_enc_act_func=list(['relu']),
                 ae_dec_act_func=list(['linear']),
                 ae_l1_reg=list([0.0]),
                 ae_l2_reg=list([0.0]),
                 ae_loss_func=list(['mse']),
                 ae_num_epochs=list([10]),
                 ae_batch_size=list([100]),
                 ae_opt=list(['adam']),
                 ae_learning_rate=list([0.001]),
                 ae_momentum=list([0.5]),
                 dropout=0.1,
                 loss_func='mse',
                 enc_act_func='linear',
                 dec_act_func='linear',
                 l1_reg=0,
                 l2_reg=0,
                 opt='adam',
                 learning_rate=0.001,
                 momentum=0.5,
                 num_epochs=100,
                 batch_size=100,
                 seed=42,
                 verbose=0):

        """
        :param layers: List containing the hidden units for each layer.
        :param ae_enc_act_func: Activation function of each autoencoder.
        :param ae_dec_act_func: Activation function of each autoencoder.
        :param ae_l1_reg: L1 weight regularization penalty of each autoencoder.
        :param ae_l2_reg: L2 weight regularization penalty of each autoencoder.
        :param ae_loss_func: Loss function of each autoencoder.
        :param ae_num_epochs: Number of epochs for training of each autoencoder.
        :param ae_batch_size: Size of mini-batch of each autoencoder.
        :param ae_opt: Optimization function of each autoencoder.
        :param ae_learning_rate: Initial learning rate of each autoencoder.
        :param ae_momentum: Momentum parameter of each autoencoder.
        :param dropout: Fraction of the finetuning hidden layers units to drop.
        :param loss_func: loss function for the finetuning step.
        :param enc_act_func: Finetuning step  hidden layers activation function.
        :param dec_act_func: Finetuning step output layer activation function.
        :param l1_reg:
        :param l2_reg:
        :param opt: Optimization function for the finetuning step.
        :param learning_rate: Learning rate for the finetuning.
        :param momentum: Momentum for the finetuning.
        :param num_epochs: Number of epochs for the finetuning.
        :param batch_size: Size of each mini-batch for the finetuning.
        :param seed: positive integer for seeding random generators. Ignored if < 0.
        :param verbose:
        """

        super().__init__(name=name,
                         layers=layers,
                         enc_act_func=enc_act_func,
                         dec_act_func=dec_act_func,
                         loss_func=loss_func,
                         l1_reg=l1_reg,
                         l2_reg=l2_reg,
                         dropout=dropout,
                         num_epochs=num_epochs,
                         batch_size=batch_size,
                         opt=opt,
                         learning_rate=learning_rate,
                         momentum=momentum,
                         seed=seed,
                         verbose=verbose)

        # Autoencoder parameters
        self.ae_type = Autoencoder
        self.ae_enc_act_func = ae_enc_act_func
        self.ae_dec_act_func = ae_dec_act_func
        self.ae_l1_reg = ae_l1_reg
        self.ae_l2_reg = ae_l2_reg
        self.ae_loss_func = ae_loss_func
        self.ae_num_epochs = ae_num_epochs
        self.ae_batch_size = ae_batch_size
        self.ae_opt = ae_opt
        self.ae_learning_rate = ae_learning_rate
        self.ae_momentum = ae_momentum

        # Finetuning parameters
        self.pretrain_params = []

        self.logger.info('Done {} __init__'.format(__class__.__name__))

    def _create_layers(self, input_shape, n_output):

        """ Create the finetuning model
        :param input_shape:
        :param n_output:
        :return: self
        """

        # Hidden layers
        for n, l in enumerate(self.layers):
            self._model.add(Dense(output_dim=l,
                                  input_shape=[input_shape[1] if n == 0 else None],
                                  weights=self.pretrain_params[n],
                                  activation=self.enc_act_func,
                                  W_regularizer=l1l2(self.l1_reg, self.l2_reg),
                                  b_regularizer=l1l2(self.l1_reg, self.l2_reg)))

            if self.dropout > 0:
                self._model.add(Dropout(p=self.dropout))

        # Output layer
        self._model.add(Dense(output_dim=n_output,
                              activation=self.dec_act_func))

    def _pretrain(self, x_train, x_valid=None):

        """ Perform unsupervised pretraining of the stack of autoencoders.
        :param x_train: training set
        :param x_valid: validation set
        :return: self
        """

        self.logger.info('Starting {} unsupervised pretraining...'.format(self.name))

        self.pretrain_params = []

        next_train = x_train
        next_valid = x_valid

        aes = self._get_autoencoders_params()

        for i, ae_params in enumerate(aes):

            self.logger.info('Pre-training layer {}'.format(i))

            ae = self.ae_type(**ae_params)

            # Pretrain a single autoencoder
            ae.fit(next_train, next_valid)

            # Get autoencoder parameters
            # params[0] = weights
            # params[1] = biases
            self.pretrain_params.append(ae.get_model_parameters()['enc'])

            # Encode the data for the next layer
            next_train = ae.transform(data=next_train)

            if x_valid:
                next_valid = ae.transform(data=next_valid)

        self.logger.info('Done {} unsupervised pretraining...'.format(self.name))

    def _get_ae_args(self):
        return {
            'enc_act_func':  self.ae_enc_act_func,
            'dec_act_func':  self.ae_dec_act_func,
            'l1_reg':        self.ae_l1_reg,
            'l2_reg':        self.ae_l2_reg,
            'loss_func':     self.ae_loss_func,
            'num_epochs':    self.ae_num_epochs,
            'batch_size':    self.ae_batch_size,
            'opt':           self.ae_opt,
            'learning_rate': self.ae_learning_rate,
            'momentum':      self.ae_momentum,
        }

    def _get_autoencoders_params(self):
        ae_args = utils.expand_args(self.layers, self._get_ae_args())
        aes_params = []

        for i, l in enumerate(self.layers):
            ae_params = {
                'name': self.name + '_ae_{}'.format(i),
                'n_hidden': l,
                'verbose': self.verbose}
            ae_params.update(dict((k, v[i]) for k, v in ae_args.items()))
            aes_params.append(ae_params)

        return aes_params

    def fit(self, x_train, y_train, x_valid=None, y_valid=None):

        """
        :param x_train:
        :param y_train:
        :param x_valid:
        :param y_valid:
        :return: self
        """

        self._pretrain(x_train, x_valid)

        super().fit(x_train, y_train, x_valid, y_valid)
