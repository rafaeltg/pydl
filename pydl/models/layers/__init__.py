#!/usr/bin/env python3.7
# -*- coding: utf-8 -*-

from keras.layers.core import Dense, Dropout, SpatialDropout1D, Flatten
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM, GRU
from keras.layers.cudnn_recurrent import CuDNNLSTM, CuDNNGRU
from keras.layers.convolutional import Conv1D, Conv2D, MaxPooling1D, MaxPooling2D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from .conv_rnn import ConvLSTM1D
from .hyperopt import *


def get(identifier):

    config = {}
    class_name = ''

    if isinstance(identifier, dict):
        class_name = identifier['class_name']
        config = identifier['config']

    elif isinstance(identifier, str):
        class_name = identifier

    if class_name == 'Dense':
        return Dense(**config)

    if class_name == 'Dropout':
        return Dropout(**config)

    if class_name == 'SpatialDropout1D':
        return SpatialDropout1D(**config)

    if class_name == 'Flatten':
        return Flatten(**config)

    if class_name == 'Conv1D':
        return Conv1D(**config)

    if class_name == 'Conv2D':
        return Conv2D(**config)

    if class_name == 'MaxPooling1D':
        return MaxPooling1D(**config)

    if class_name == 'MaxPooling2D':
        return MaxPooling2D(**config)

    if class_name == 'LSTM':
        return LSTM(**config)

    if class_name == 'GRU':
        return GRU(**config)

    if class_name == 'CuDNNLSTM':
        return CuDNNLSTM(**config)

    if class_name == 'CuDNNGRU':
        return CuDNNGRU(**config)

    if class_name == 'ConvLSTM1D':
        return ConvLSTM1D(**config)

    if class_name == 'ConvLSTM2D':
        return ConvLSTM2D(**config)

    if class_name == 'TimeDistributed':
        return TimeDistributed(**config)

    if class_name == 'TimeDistributed':
        return BatchNormalization(**config)

    raise ValueError("Invalid identifier '{}'".format(class_name))
