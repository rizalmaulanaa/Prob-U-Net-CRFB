import numpy as np
import tensorflow as tf

from keras import backend as K
from tensorflow.keras.metrics import Mean
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import UpSampling2D, MaxPool2D, Add, Subtract
from tensorflow.keras.layers import Concatenate, Reshape, Dense, Multiply
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation

def handle_block_names(stage, cols, type_='decoder', type_act='relu'):
    conv_name = '{}_stage{}-{}_conv'.format(type_, stage, cols)
    bn_name = '{}_stage{}-{}_bn'.format(type_, stage, cols)
    act_name = '{}_stage{}-{}_relu'.format(type_, stage, cols)
    up_name = '{}_stage{}-{}_upat'.format(type_, stage, cols)
    add_name = '{}_stage{}-{}_add'.format(type_, stage, cols)
    sigmoid_name = '{}_stage{}-{}_sigmoid'.format(type_, stage, cols)
    mul_name = '{}_stage{}-{}_mul'.format(type_, stage, cols)
    merge_name = 'merge_{}-{}'.format(stage, cols)

    return conv_name, bn_name, act_name, up_name, merge_name, add_name, sigmoid_name, mul_name

def conv_block(filters, stage, cols, kernel_size=3, use_batchnorm=True,
               amount=3, type_act='relu', type_block='encoder'):

    def layer(x):
        act_function = tf.identity if type_act == 'identity' else type_act
        conv_name, bn_name, act_name, _, _, _, _, _ = handle_block_names(stage, cols, type_=type_block, type_act=type_act)
        for i in range(amount):
            temp = '_'+str(i+1)
            x = ConvRelu(filters, kernel_size=kernel_size, use_batchnorm=use_batchnorm,
                          conv_name=conv_name+temp, bn_name=bn_name+temp,
                          act_name=act_name+temp, act_function=act_function) (x)
        return x
    return layer

def z_mu_sigma(filters, stage, cols, use_batchnorm=True, type_block='z'):
    def layer(x):
        mu = conv_block(filters, stage, cols, use_batchnorm=use_batchnorm, amount=1,
                        kernel_size=1, type_act='identity', type_block='mu') (x)
        sigma = conv_block(filters, stage, cols, use_batchnorm=use_batchnorm, amount=1,
                           kernel_size=1, type_act='softplus', type_block='sigma') (x)

        z = Multiply(name='z_stage{}-{}_mul'.format(stage,cols)) ([
            sigma, tf.random.normal(tf.shape(mu), 0, 1, dtype=tf.float32)])
        z = Add(name='z_stage{}-{}_add'.format(stage,cols)) ([mu, z])
        return z, mu, sigma
    return layer

def res_block(filters, stage, cols, kernel_size=3, use_batchnorm=True,
              amount=3, type_act='relu', type_block='encoder'):
    def layer(x):
        for k in range(amount):
            block_num = str(cols)+'_'+str(k)
            x = res_connection(filters, stage, block_num, kernel_size=kernel_size,
                               type_block=type_block, activation_fn=type_act) (x)

        return x
    return layer

def inception_block(filters, stage, cols, kernel_size=3, use_batchnorm=True,
                    amount=3, type_act='relu', type_block='encoder'):
    def layer(x):
        kernel_size_list = [1,3,5,7]
        net = [None] * (len(kernel_size_list) + 1)
        conv_name, bn_name, act_name, _, merge_name, _, _, _ = handle_block_names(stage, cols, type_=type_block, type_act=type_act)

        for num in range(amount):
            for k,i in enumerate(kernel_size_list):
                temp = '1_{}_{}'.format(str(k+1), num+1)
                net[k] = ConvRelu(filters, kernel_size=1, use_batchnorm=use_batchnorm,
                                  conv_name=conv_name+temp, bn_name=bn_name+temp,
                                  act_name=act_name+temp, act_function=type_act) (x)

                if i != 1:
                    temp = '_{}_{}'.format(str(i), num+1)
                    net[k] = ConvRelu(filters, kernel_size=i, use_batchnorm=use_batchnorm,
                                      conv_name=conv_name+temp, bn_name=bn_name+temp,
                                      act_name=act_name+temp, act_function=type_act) (net[k])

            temp = '_afpool'+str(num+1)
            x = MaxPool2D(pool_size=3, strides=1, padding='same', name='{}_stage{}-{}_pool_in_{}'.
                          format(type_block, stage, cols, num+1)) (x)
            net[-1] = ConvRelu(filters, kernel_size=1, use_batchnorm=use_batchnorm,
                               conv_name=conv_name+temp, bn_name=bn_name+temp,
                               act_name=act_name+temp, act_function=type_act) (x)

            x = Concatenate(name=merge_name+'_'+str(num+1)) (net)
        return x
    return layer

def CRFB(filters, stage, cols, down_size, amount=2, kernel_size=3,
         use_batchnorm=True, type_act='relu', type_block='CRFB'):
    def layer(x_d, x_u):
        _, _, _, up_name, _, add_name, _, _ = handle_block_names(stage, cols, type_=type_block, type_act=type_act)

        temp_x_d = x_d; temp_x_u = x_u
        x_d = conv_block(filters, 1, cols, amount=amount,
                         type_block=type_block, use_batchnorm=use_batchnorm) (temp_x_u)
        x_d = MaxPool2D(pool_size=down_size**2, name='CRFB_stage1-{}_pool'.format(cols)) (x_d)
        x_d = Add(name=add_name+'_0') ([x_d, temp_x_d])

        x_u = conv_block(filters, 0, cols, amount=amount,
                         type_block=type_block, use_batchnorm=use_batchnorm) (temp_x_d)
        x_u = UpSampling2D(size=down_size**2, name=up_name) (x_u)
        x_u = Add(name=add_name+'_1') ([x_u, temp_x_u])

        return x_d, x_u
    return layer
