#!/usr/bin/env python3
#-*- coding:utf-8 -*-
import tensorflow as tf

# Smooth L1 loss function
def smooth_L1_loss(x, beta=1.0, reduction='mean'):
    '''
    x =(y_true-y_pred)
    The data must have ground truth.
    '''
    # 0.5 * (x**2)
    mul = tf.math.multiply(0.5, tf.math.pow(x, 2.0)) / beta
    # |x|-0.5
    sub = tf.math.substract(tf.math.abs(x), 0.5) * beta
    # list of |x| < 1.0
    cond = tf.math.less(tf.math.abs(x), 1.0)
    out = tf.where(cond, mul, sub)

    if reduction == 'sum':
        out = tf.math.reduce_sum(out)
    else:
        out = tf.math.reduce_mean(out)

    return out
