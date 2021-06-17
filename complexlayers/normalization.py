#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Time      : 2017/12/29 16:27
# Author    : zsh_o

import numpy as np
from keras.layers import Layer, InputSpec
from keras import initializers, regularizers, constraints
from . import initializers as cinitializers
import keras.backend as K


def sqrt(shape):       # 构造一个全根号2分之1矩阵

    value = (1 / np.sqrt(2.0)) * K.ones(shape)
    return value


def initGet(init):
    if init in ['sqrt']:
        return sqrt
    else:
        return initializers.get(init)


def initSet(init):
    if init in [sqrt]:
        return 'sqrt'
    else:
        return initializers.serialize(init)


class ComplexBatchNormalization(Layer):
    # keras官方batchnormalization在这里加了一个broadcasting，不是很明白为啥要加，就没加
    # 把计算inference的所有函数转到class里面
    # 复数相关的层，复数的输入输出为两个张量，real part and image part, 对应的input and output shape 也为两个
    def __init__(self,
                 axis=-1,  # 在所有维度进行batch_norm
                 momentum=0.9,
                 epsilon=1e-4,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_diag_initializer='sqrt',  # gamma is matrix with freedom of three degrees: rr, ri, ii
                 gamma_off_initializer='zeros',
                 moving_mean_initializer='zeros',  # 三个moving_average变量均不可训练，用于计算和保存均值和协方差矩阵
                 moving_variance_initializer='sqrt',  # 每次计算该batch的均值和协方差矩阵，然后用加动量的moving_average更新moving_mean, moving_var, moving_cov
                 moving_covariance_initializer='zeros',
                 beta_regularizer=None,
                 gamma_diag_regularizer=None,  # 正则化
                 gamma_off_regularizer=None,
                 beta_constraint=None,  # 约束
                 gamma_diag_constraint=None,
                 gamma_off_constraint=None,
                 trainable=True,
                 name=None,
                 **kwargs):
        super(ComplexBatchNormalization, self).__init__(trainable=trainable, name=name, **kwargs)
        self.supports_masking = True
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer               = cinitializers.get(beta_initializer)
        self.gamma_diag_initializer         = cinitializers.get(gamma_diag_initializer)
        self.gamma_off_initializer          = cinitializers.get(gamma_off_initializer)
        self.moving_mean_initializer        = cinitializers.get(moving_mean_initializer)
        self.moving_variance_initializer    = cinitializers.get(moving_variance_initializer)
        self.moving_covariance_initializer  = cinitializers.get(moving_covariance_initializer)
        self.beta_regularizer               = regularizers.get(beta_regularizer)
        self.gamma_diag_regularizer         = regularizers.get(gamma_diag_regularizer)
        self.gamma_off_regularizer          = regularizers.get(gamma_off_regularizer)
        self.beta_constraint                = constraints.get(beta_constraint)
        self.gamma_diag_constraint          = constraints.get(gamma_diag_constraint)
        self.gamma_off_constraint           = constraints.get(gamma_off_constraint)

    def build(self, input_shape):
        input_shapes = input_shape
        # assert(input_shapes[0] == input_shapes[1])
        input_shape = input_shapes[0]
        dim = input_shape[self.axis]   # tuple[-1]返回shape的最后一位
        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                                                        'input tensor should have a defined dimension '
                                                        'but the layer received an input with shape ' +
                             str(input_shape) + '.')
        # self.input_spec = InputSpec(ndim=len(input_shape), axes={self.axis: dim})
        shape = (dim,)       # 用于对实数的初始化                ?????????????????????????????????????????????????????
        if self.scale:
            self.gamma_rr = self.add_weight(shape=shape,     # shape ???????????????????????????????????????????????
                                            name='gamma_rr',
                                            initializer=self.gamma_diag_initializer,
                                            regularizer=self.gamma_diag_regularizer,
                                            constraint=self.gamma_diag_constraint)
            self.gamma_ii = self.add_weight(shape=shape,
                                            name='gamma_ii',
                                            initializer = self.gamma_diag_initializer,
                                            regularizer = self.gamma_diag_regularizer,
                                            constraint = self.gamma_diag_constraint)
            self.gamma_ri = self.add_weight(shape = shape,
                                            name = 'gamma_ri',
                                            initializer = self.gamma_off_initializer,
                                            regularizer = self.gamma_off_regularizer,
                                            constraint = self.gamma_off_constraint)
            self.moving_Vrr = self.add_weight(shape = shape,
                                              initializer = self.moving_variance_initializer,
                                              name = 'moving_Vrr',
                                              trainable = False)
            self.moving_Vii = self.add_weight(shape = shape,
                                              initializer = self.moving_variance_initializer,
                                              name = 'moving_Vii',
                                              trainable = False)
            self.moving_Vri = self.add_weight(shape = shape,
                                              initializer = self.moving_covariance_initializer,
                                              name = 'moving_Vri',
                                              trainable = False)
        else:
            self.gamma_rr = None
            self.gamma_ii = None
            self.gamma_ri = None
            self.moving_Vrr = None
            self.moving_Vii = None
            self.moving_Vri = None

        if self.center:
            self.beta_real = self.add_weight(shape = shape,
                                             name = 'beta_real',
                                             initializer = self.beta_initializer,
                                             regularizer = self.beta_regularizer,
                                             constraint = self.beta_constraint)
            self.beta_image = self.add_weight(shape = shape,
                                              name = 'beta_image',
                                              initializer = self.beta_initializer,
                                              regularizer = self.beta_regularizer,
                                              constraint = self.beta_constraint)
            self.moving_mean_real = self.add_weight(shape = shape,
                                                    initializer = self.moving_mean_initializer,
                                                    name = 'moving_mean_real',
                                                    trainable = False)
            self.moving_mean_image = self.add_weight(shape = shape,
                                                     initializer = self.moving_mean_initializer,
                                                     name = 'moving_mean_image',
                                                     trainable = False)
        else:
            self.beta_real = None
            self.beta_image = None
            self.moving_mean_real = None
            self.moving_mean_image = None

        self.built = True

    def call(self, inputs, training=None):
        assert isinstance(inputs, list)
        input_real, input_image = inputs[0], inputs[1]
        input_shape = K.int_shape(input_real)
        ndim = len(input_shape)       # 输入数据维度
        reduction_axes = list(range(ndim))  # 计算均值的时候需要指定维度
        del reduction_axes[self.axis]       # 删除最后一位
        mu_real = K.mean(input_real, axis=reduction_axes)  # 复数加减不涉及实虚部转换   #?????????????????????????????
        mu_image = K.mean(input_image, axis=reduction_axes)                          #????????????????????????????

        # center_x = x - E[x]   是否归一化
        if self.center:
            centered_real = input_real - mu_real
            centered_image = input_image - mu_image
        else:
            centered_real = input_real
            centered_image = input_image
        centered_squared_real = centered_real ** 2     # 乘方
        centered_squared_image = centered_image ** 2   #
        centered = K.concatenate([centered_real, centered_image])
        centered_squared = K.concatenate([centered_squared_real, centered_squared_image])

        # 计算V矩阵
        if self.scale:
            Vrr = K.mean(              # Vrr = Cov(R(x), R(x)), Cov(X, Y) = E((X - E(X))(Y - E(Y)))
                centered_squared_real,
                axis=reduction_axes,
            ) + self.epsilon
            Vii = K.mean(              # Vii = Cov(I(x), I(x))
                centered_squared_image,
                axis=reduction_axes,
            ) + self.epsilon
            Vri = K.mean(              # Vri = Cov(R(x), I(x))， Vri == Vir
                centered_real * centered_image,
                axis=reduction_axes,
            )
        elif self.center:
            Vrr = None
            Vii = None
            Vri = None
        else:
            raise ValueError('Error. Both scale and center in batchnorm are set to False.')
        input_bn = self.complexBN(centered_real, centered_image, Vrr, Vii, Vri)
        if training in {0, False}:
            return input_bn
        else:
            update_list = []
            if self.center:
                update_list.append(K.moving_average_update(self.moving_mean_real, mu_real, self.momentum))
                update_list.append(K.moving_average_update(self.moving_mean_image, mu_image, self.momentum))
            if self.scale:
                update_list.append(K.moving_average_update(self.moving_Vrr, Vrr, self.momentum))
                update_list.append(K.moving_average_update(self.moving_Vii, Vii, self.momentum))
                update_list.append(K.moving_average_update(self.moving_Vri, Vri, self.momentum))
            self.add_update(update_list, inputs)

            def normalize_inference():
                if self.center:
                    inference_centered_real = input_real - self.moving_mean_real
                    inference_centered_image = input_image - self.moving_mean_image
                else:
                    inference_centered_real = input_real
                    inference_centered_image = input_image
                return self.complexBN(inference_centered_real, inference_centered_image, Vrr, Vii, Vri)

        return K.in_train_phase(input_bn,            # train阶段使用input_bn, 其他时候使用normalize_inference
                                normalize_inference,
                                training=training)

    # 计算V的-1/2次方（W）再乘（x - E（x））
    def complex_std(self, centered_real, centered_image, Vrr, Vii, Vri):
        # sqrt of a 2x2 matrix, https://en.wikipedia.org/wiki/Square_root_of_a_2_by_2_matrix
        tau = Vrr + Vii
        delta = (Vrr * Vii) - (Vri ** 2)
        s = K.sqrt(delta)
        t = K.sqrt(tau + 2 * s)

        # matrix inverse, http://mathworld.wolfram.com/MatrixInverse.html
        inverse_st = 1.0 / (s * t)
        Wrr = (Vii + s) * inverse_st
        Wii = (Vrr + s) * inverse_st
        Wri = -Vri * inverse_st

        output_real = Wrr * centered_real + Wri * centered_image  # 中心化的x（real， image）以列形式与W相乘
        output_image = Wri * centered_real + Wii * centered_image

        return output_real, output_image

    # 计算BN，乘gamma矩阵，再加上beta
    def complexBN(self, centered_real, centered_image, Vrr, Vii, Vri):
        output_real = centered_real
        output_image = centered_image
        if self.scale:
            t_real, t_image = self.complex_std(centered_real, centered_image, Vrr, Vii, Vri)
            output_real = self.gamma_rr * t_real + self.gamma_ri * t_image
            output_image = self.gamma_ri * t_real + self.gamma_ii * t_image
        if self.center:
            output_real = output_real + self.beta_real
            output_image = output_image + self.beta_image

        return output_real, output_image

    def get_config(self):
        config = {
            'axis': self.axis,
            'momentum': self.momentum,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer':                 cinitializers.serialize(self.beta_initializer),
            'gamma_diag_initializer':           cinitializers.serialize(self.gamma_diag_initializer),
            'gamma_off_initializer':            cinitializers.serialize(self.gamma_off_initializer),
            'moving_mean_initializer':          cinitializers.serialize(self.moving_mean_initializer),
            'moving_variance_initializer':      cinitializers.serialize(self.moving_variance_initializer),
            'moving_covariance_initializer':    cinitializers.serialize(self.moving_covariance_initializer),
            'beta_regularizer':                   regularizers.serialize(self.beta_regularizer),
            'gamma_diag_regularizer':            regularizers.serialize(self.gamma_diag_regularizer),
            'gamma_off_regularizer':             regularizers.serialize(self.gamma_off_regularizer),
            'beta_constraint':                    constraints.serialize(self.beta_constraint),
            'gamma_diag_constraint':              constraints.serialize(self.gamma_diag_constraint),
            'gamma_off_constraint':               constraints.serialize(self.gamma_off_constraint),
        }
        base_config = super(ComplexBatchNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
