from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pydl.models.autoencoder_models.denoising_autoencoder import DenoisingAutoencoder
from pydl.models.autoencoder_models.stacked_autoencoder import StackedAutoencoder


class StackedDenoisingAutoencoder(StackedAutoencoder):

    """ Implementation of Stacked Denoising Autoencoders using TensorFlow.
    """

    def __init__(self,
                 name='sdae',
                 layers=list([64, 32]),
                 ae_enc_act_func=list(['relu']),
                 ae_dec_act_func=list(['linear']),
                 ae_l1_reg=list([0.0]),
                 ae_l2_reg=list([0.0]),
                 ae_loss_func=list(['mse']),
                 ae_num_epochs=list([10]),
                 ae_batch_size=list([100]),
                 ae_opt=list(['adam']),
                 ae_learning_rate=list([0.01]),
                 ae_momentum=list([0.5]),
                 ae_corr_type=list(['gaussian']),
                 ae_corr_param=list([0.2]),
                 dropout=0.1,
                 loss_func='mse',
                 enc_act_func='relu',
                 dec_act_func='linear',
                 l1_reg=0,
                 l2_reg=0,
                 opt='adam',
                 learning_rate=0.001,
                 momentum=0.5,
                 num_epochs=10,
                 batch_size=100,
                 seed=42,
                 verbose=0):

        """
        :param layers: list containing the hidden units for each layer
        :param ae_enc_act_func: Activation function for the encoder.
        :param ae_dec_act_func: Activation function for the decoder.
        :param ae_l1_reg: L1 weight regularization penalty of each denoising autoencoder.
        :param ae_l2_reg: L2 weight regularization penalty of each denoising autoencoder.
        :param ae_loss_func: Loss function.
        :param ae_num_epochs: Number of epochs for training.
        :param ae_batch_size: Size of each mini-batch.
        :param ae_opt: Optimizer to use. string, default 'gradient_descent'.
        :param ae_learning_rate: Initial learning rate.
        :param ae_momentum: 'Momentum parameter.
        :param ae_corr_type: type of input corruption. ["masking", "gaussian"]
        :param ae_corr_param: 'scale' parameter for Aditive Gaussian Corruption ('gaussian') or
                           'keep_prob' parameter for Masking Corruption ('masking')
        :param dropout: hidden layers dropout parameter.
        :param loss_func: Cost function for the fine tunning step.
        :param enc_act_func: finetuning step hidden layers activation function.
        :param dec_act_func: finetuning step output layer activation function.
        :param l1_reg:
        :param l2_reg:
        :param opt: optimizer for the finetuning phase
        :param learning_rate: learning rate for the finetuning.
        :param momentum: momentum for the finetuning.
        :param num_epochs: Number of epochs for the finetuning.
        :param batch_size: Size of each mini-batch for the finetuning.
        :param seed: positive integer for seeding random generators. Ignored if < 0.
        """

        super().__init__(name=name,
                         layers=layers,
                         ae_enc_act_func=ae_enc_act_func,
                         ae_dec_act_func=ae_dec_act_func,
                         ae_l1_reg=ae_l1_reg,
                         ae_l2_reg=ae_l2_reg,
                         ae_loss_func=ae_loss_func,
                         ae_num_epochs=ae_num_epochs,
                         ae_batch_size=ae_batch_size,
                         ae_opt=ae_opt,
                         ae_learning_rate=ae_learning_rate,
                         ae_momentum=ae_momentum,
                         dropout=dropout,
                         loss_func=loss_func,
                         enc_act_func=enc_act_func,
                         dec_act_func=dec_act_func,
                         l1_reg=l1_reg,
                         l2_reg=l2_reg,
                         opt=opt,
                         learning_rate=learning_rate,
                         momentum=momentum,
                         num_epochs=num_epochs,
                         batch_size=batch_size,
                         seed=seed,
                         verbose=verbose)

        # Denoising Autoencoder parameters
        self.ae_type = DenoisingAutoencoder
        self.ae_corr_type = ae_corr_type
        self.ae_corr_param = ae_corr_param

        self.logger.info('Done {} __init__'.format(__class__.__name__))

    def _get_ae_args(self):
        ae_args = super()._get_ae_args()
        ae_args['corr_type'] = self.ae_corr_type
        ae_args['corr_param'] = self.ae_corr_param
        return ae_args
