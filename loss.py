#!/usr/bin/env python3
#-*- coding:utf-8 -*-
import tensorflow as tf

# Smooth L1 loss function
def smooth_L1_loss(x):
    '''
    x =(y_true-y_pred)
    Both y_true and y_pred is the positive position
    '''
    # 0.5 * (x**2)
    mul = tf.math.multiply(0.5, tf.pow(x, 2.0))
    # |x|-0.5
    sub = tf.math.substract(tf.math.abs(x), 0.5)
    # list of |x| < 1.0
    cond = tf.math.less(tf.math.abs(x), 1.0)
    out = tf.where(cond, mul, sub)
    return tf.math.reduce_mean(out)
