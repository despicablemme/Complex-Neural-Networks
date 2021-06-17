"""
2021.6.16 16:14
修复complex conv1d
修复real与image初始化器问题
2d 3d未尝试
"""

import numpy as np
import functools
import six
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import activations, regularizers, constraints
from . import initializers as cinitializers
from tensorflow.python.keras.layers import Layer, InputSpec
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.ops import nn_ops
import tensorflow as tf         # for spectral parametrization


class _ComplexConv(Layer):
    """
    论文源代码中这个地方外加了两个功能，都是针对权值的
      1、spectral parametrization: 表示权值是在谱域定义的，进行卷积之前需要把其用IFFT转变为原始的空域
      2、normalize weight: 每一次进行卷积之前都把权值归一化，采用的是bn里面定义的complex_normalization，这也应该是在complexBN中作者没有把那两个函数放到类里面的原因吧
    关于1，和pooling里面的那个SpectralPooling均是采用论文：Spectral Representations for Convolutional Neural Networks（https://arxiv.org/abs/1506.03767）
          卷积的权值在频率域表示，进行卷积之前需要用IFFT把其反变换为空域，论文中显示，这种表示方法学习到的权值更稀疏更少（学习改变的权值的量少）
    关于2，个人感觉没什么用，在这个地方直接对权值进行操作不如加相似功能的正则化

    卷积网络里面越倾向高层，卷积层输出得到的特征谱越稀疏
    保留1，舍弃2

    输入方式：input:[x1, x2], output:[y1, y2], 以列表形式输入输出
    """
    def __init__(self,
                 rank,
                 filters,          # 卷积输出滤波器的数目，输出的特征的数目，输出的channel数目
                 kernel_size,      # kernel_size为tuple，长度为rank，单个卷积核的大小
                 strides=1,        # 步长
                 padding='valid',  # 填充模式
                 data_format='channels_last',
                 dilation_rate=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='he_complex',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 seed=None,
                 spectral_parametrization=False,
                 name=None,
                 trainable=True,
                 **kwargs):
        super(_ComplexConv, self).__init__(trainable=trainable, name=name, **kwargs)
        self.rank = rank
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, rank, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, rank, 'dilation_rate')
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        # self.kernel_initializer = cinitializers.get(kernel_initializer)
        self.kernel_initializer_real = cinitializers.get(kernel_initializer+'_real')
        self.kernel_initializer_image = cinitializers.get(kernel_initializer + '_image')
        self.bias_initializer = cinitializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        # self.input_spec = InputSpec(ndim=self.rank + 2)
        self.spectral_parametrization = spectral_parametrization
        self.seed = seed if seed is not None else np.random.randint(1, 1e6)

    def build(self, input_shape):        # input_shape传入一个列表
        input_shapes = input_shape
        # assert (input_shapes[0] == input_shapes[1])
        input_shape = input_shapes[0]
        assert len(input_shape) >= 2

        if self.data_format == 'channels_first':    # 默认使用else模式，channel在最后,否则在第二位
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_channel = input_shape[channel_axis]      # 确认channel的数目
        # 基于tensorflow，conv2d的卷积核的shape=(kernel_size(n*n),in_channel, out_channel)
        kernel_shape = self.kernel_size + (input_channel, self.filters)
        self.kernel_shape = kernel_shape
        kernel_shapes = kernel_shape
        self.kernel_real = self.add_weight(shape=kernel_shapes,
                                           initializer=self.kernel_initializer_real,
                                           name='kernel',
                                           regularizer=self.kernel_regularizer,
                                           constraint=self.kernel_constraint)
        self.kernel_image = self.add_weight(shape=kernel_shapes,
                                            initializer=self.kernel_initializer_image,
                                            name='kernel',
                                            regularizer=self.kernel_regularizer,
                                            constraint=self.kernel_constraint)
        # self.kernel_real = self.kernel[0]
        # self.kernel_image = self.kernel[1]
        self.kernel_complex = tf.complex(self.kernel_real, self.kernel_image)    # 留待下面做spectral transform

        if self.use_bias:
            self.bias_real = self.add_weight(shape=(self.filters,),
                                             initializer=self.bias_initializer,
                                             name='bias_real',
                                             regularizer=self.bias_regularizer,
                                             constraint=self.bias_constraint)
            self.bias_image = self.add_weight(shape=(self.filters,),
                                              initializer=self.bias_initializer,
                                              name='bias_image',
                                              regularizer=self.bias_regularizer,
                                              constraint=self.bias_constraint)
        else:
            self.bias_real = None
            self.bias_image = None

        # Convert Keras formats to TF native formats. 从keras的conv中抄过来,使用self._convolution_op计算卷积运算
        self._tf_data_format = conv_utils.convert_data_format(
            self.data_format, self.rank + 2)
        if self.padding == 'causal':
            tf_padding = 'VALID'  # Causal padding handled in `call`.
        elif isinstance(self.padding, six.string_types):
            tf_padding = self.padding.upper()
        else:
            tf_padding = self.padding
        tf_dilations = list(self.dilation_rate)
        tf_strides = list(self.strides)
        tf_op_name = self.__class__.__name__
        if tf_op_name == 'ComplexConv1D':     # tf_op_name对2d与3d的conv可能有影响
            tf_op_name = 'conv1d'             # Backwards compat.
        self._convolution_op = functools.partial(
            nn_ops.convolution_v2,
            strides=tf_strides,
            padding=tf_padding,
            dilations=tf_dilations,
            data_format=self._tf_data_format,
            name=tf_op_name)

        self.built = True

        self.convArgs = {            # 卷积操作对应的参数，以字典传入
            'strides':          self.strides[0] if self.rank == 1 else self.strides,
            'padding':          self.padding,
            'data_format':      self.data_format,
            'dilation_rate':    self.dilation_rate[0] if self.rank == 1 else self.dilation_rate
        }
        self.convFunc = {
            1:  K.conv1d,
            2:  K.conv2d,
            3:  K.conv3d
        }[self.rank]        # 选择rank对应的
        self.ifftFunc = {
            1:  tf.signal.ifft,
            2:  tf.signal.ifft2d,
            3:  tf.signal.ifft3d
        }[self.rank]

    def call(self, inputs, **kwargs):
        assert isinstance(inputs, list)
        input_real, input_image = inputs[0], inputs[1]

        # transform weights to spectral domain
        kernel_complex = self.kernel_complex      # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%暂时不懂
        if self.spectral_parametrization:         # 频域转换,###################################
            flat_shape = (self.kernel_shape[-1] * self.kernel_shape[-2],) + self.kernel_shape[:-2]
            fk = K.reshape(kernel_complex, flat_shape)
            fk = self.ifftFunc(fk)
            kernel_complex = K.reshape(fk, self.kernel_shape)
        kernel_real = tf.math.real(kernel_complex)
        kernel_image = tf.math.imag(kernel_complex)

        output_real = self._convolution_op(input_real, kernel_real) - self._convolution_op(input_image, kernel_image)
        output_image = self._convolution_op(input_real, kernel_image) - self._convolution_op(input_image, kernel_real)
        # output_real = self.convFunc(input_real, kernel_real, **self.convArgs)
        #               - self.convFunc(input_image, kernel_image, **self.convArgs)
        # output_image = self.convFunc(input_real, kernel_image, **self.convArgs)
        #               + self.convFunc(input_image, kernel_real, **self.convArgs)
        if self.use_bias:
            output_real = K.bias_add(output_real, self.bias_real)
            output_image = K.bias_add(output_image, self.bias_image)
        if self.activation is not None:
            output_real = self.activation(output_real)
            output_image = self.activation(output_image)
        return output_real, output_image

    def compute_output_shape(self, input_shape):
        input_shapes = input_shape
        # assert (input_shapes[0] == input_shapes[1])
        input_shape = input_shapes[0]
        if self.data_format == 'channels_last':
            space = input_shape[1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            single_shape = (input_shape[0],) + tuple(new_space) + (self.filters,)
        elif self.data_format == 'channels_first':
            space = input_shape[2:]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            single_shape = (input_shape[0], self.filters) + tuple(new_space)
        else:
            raise ValueError('Invalid data format: ' + self.data_format)
        return [single_shape] * 2

    def get_config(self):
        config = {
            'rank':                 self.rank,
            'filters':              self.filters,
            'kernel_size':          self.kernel_size,
            'strides':              self.strides,
            'padding':              self.padding,
            'data_format':          self.data_format,
            'dilation_rate':        self.dilation_rate,
            'activation':           activations.serialize(self.activation),
            'use_bias':             self.use_bias,
            'kernel_initializer':   cinitializers.serialize(self.kernel_initializer),
            'bias_initializer':     cinitializers.serialize(self.bias_initializer),
            'kernel_regularizer':   regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer':     regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint':    constraints.serialize(self.kernel_constraint),
            'bias_constraint':      constraints.serialize(self.bias_constraint),
            'spectral_parametrization': self.spectral_parametrization,
            'seed':                     self.seed,
        }
        base_config = super(_ComplexConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ComplexConv1D(_ComplexConv):
    # 一些不必要的相同的参数可以用**kwargs来传递，不知道为啥都要重写一遍，keras官方文档也是这样的
    def __init__(self,
                 filters,
                 kernel_size,
                 strides = 1,
                 padding = 'valid',
                 dilation_rate = 1,
                 activation = None,
                 use_bias = True,
                 kernel_initializer = 'he_complex',
                 bias_initializer = 'zeros',
                 kernel_regularizer = None,
                 bias_regularizer = None,
                 activity_regularizer = None,
                 kernel_constraint = None,
                 bias_constraint = None,
                 spectral_parametrization = False,
                 **kwargs):
        super(ComplexConv1D, self).__init__(
            rank = 1,
            filters = filters,
            kernel_size = kernel_size,
            strides = strides,
            padding = padding,
            dilation_rate = dilation_rate,
            activation = activation,
            use_bias = use_bias,
            kernel_initializer = kernel_initializer,
            bias_initializer = bias_initializer,
            kernel_regularizer = kernel_regularizer,
            bias_regularizer = bias_regularizer,
            activity_regularizer = activity_regularizer,
            kernel_constraint = kernel_constraint,
            bias_constraint = bias_constraint,
            spectral_parametrization = spectral_parametrization,
            **kwargs
        )

    def get_config(self):
        config = super(ComplexConv1D, self).get_config()
        config.pop('rank')
        config.pop('data_format')
        return config


class ComplexConv2D(_ComplexConv):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 dilation_rate=(1, 1),
                 activation=None,
                 use_bias=True,
                 kernel_initializer='he_complex',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 spectral_parametrization=False,
                 **kwargs):
        super(ComplexConv2D, self).__init__(
            rank = 2,
            filters = filters,
            kernel_size = kernel_size,
            strides = strides,
            padding = padding,
            dilation_rate = dilation_rate,
            activation = activation,
            use_bias = use_bias,
            kernel_initializer = kernel_initializer,
            bias_initializer = bias_initializer,
            kernel_regularizer = kernel_regularizer,
            bias_regularizer = bias_regularizer,
            activity_regularizer = activity_regularizer,
            kernel_constraint = kernel_constraint,
            bias_constraint = bias_constraint,
            spectral_parametrization = spectral_parametrization,
            **kwargs
        )

    def get_config(self):
        config = super(ComplexConv2D, self).get_config()
        config.pop('rank')
        config.pop('data_format')
        return config

class ComplexConv3D(_ComplexConv):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides = (1, 1, 1),
                 padding = 'valid',
                 dilation_rate = (1, 1, 1),
                 activation = None,
                 use_bias = True,
                 kernel_initializer = 'he_complex',
                 bias_initializer = 'zeros',
                 kernel_regularizer = None,
                 bias_regularizer = None,
                 activity_regularizer = None,
                 kernel_constraint = None,
                 bias_constraint = None,
                 spectral_parametrization = False,
                 **kwargs):
        super(ComplexConv3D, self).__init__(
            rank = 3,
            filters = filters,
            kernel_size = kernel_size,
            strides = strides,
            padding = padding,
            dilation_rate = dilation_rate,
            activation = activation,
            use_bias = use_bias,
            kernel_initializer = kernel_initializer,
            bias_initializer = bias_initializer,
            kernel_regularizer = kernel_regularizer,
            bias_regularizer = bias_regularizer,
            activity_regularizer = activity_regularizer,
            kernel_constraint = kernel_constraint,
            bias_constraint = bias_constraint,
            spectral_parametrization = spectral_parametrization,
            **kwargs
        )

    def get_config(self):
        config = super(ComplexConv3D, self).get_config()
        config.pop('rank')
        config.pop('data_format')
        return config