#!/usr/bin/env python3
#-*- coding:utf-8 -*-
import tensorflow as tf
from tensorflow.keras.losses import Loss

# Smooth L1 loss function
class SmoothL1Loss(Loss):
    def __init__(self, beta=1.0, name='smooth_1l_loss', reduction='mean'):
        self.beta = 1.0
        self.name = name
        self.reduction = reduction

    def call(self, y_true, y_pred):
        error     = tf.substract(y_pred, y_true)
        abs_error = tf.abs(error)
        half      = tf.convert_to_tensor(0.5, dtype=abs_error.dtype)

        loss =  tf.where(
                abs_error < self.beta,
                (half * tf.square(error)) / self.beta,
                abs_error - half * self.beta
            )

        if self.reduction == 'mean':
            return tf.reduce_mean(loss, axis=-1)
        else:
            return tf.reduce_sum(loss, axis=-1)
