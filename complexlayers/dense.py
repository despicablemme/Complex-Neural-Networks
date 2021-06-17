"""
2021.6.16 16.28
修复kernel_shape问题
"""
from keras import backend as K
from keras import activations, initializers, regularizers, constraints
from . import initializers as cinitializers
from keras.layers import Layer, InputSpec
import numpy as np


class ComplexDense(Layer):
    """
    z = x + iy, W = A + iB, b = br + ibi
    output = Wz + b
           = (Ax - By + br) + i(Ay + Bx + bi)
    init_shape:
          W: (2 x (input_shape) x )
    """

    def __init__(self,
                 units,
                 activation=None,
                 use_bias=True,
                 # init_criterion = 'he',          # keras有一个设计不合理的地方，initializer传入的只能是不带参数的可执行的函数体
                 kernel_initializer='he_complex',  # 但当继承自initializer需要带参数的时候就不能用了，只能是需要先定义该initializer类，用这样的方法把参数传进去
                 bias_initializer='zeros',         # 恩，其实也可以在initializers类文件里面，把所有的情况都列出来。。。现使用这种方法
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 seed=None,
                 name=None,
                 trainable=True,
                 **kwargs):
        super(ComplexDense, self).__init__(trainable=trainable, name=name, **kwargs)
        self.output_units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        # self.kernel_initializer = cinitializers.get(kernel_initializer)
        self.kernel_initializer_real = cinitializers.get(kernel_initializer+'_real')
        self.kernel_initializer_image = cinitializers.get(kernel_initializer+'_image')
        self.bias_initializer = cinitializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        if seed is None:
            self.seed = np.random.randint(1, 1e6)
        else:
            self.seed = seed
        # self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    # build：创建层的权重，用self.add_weight自定义并添加权重矩阵
    def build(self, input_shape):       # input_shape来自Layer已处理好， input_shape是一个列表[shape_I,shape_Q]
        input_shapes = input_shape      # dense层input_shape:(batch_size, units)
        input_shape = input_shapes[0]   # 单独一路的input_shape
        assert len(input_shape) >= 2    # 确保是二维
        input_units = input_shape[-1]     # shape的最后一位，即dense输入节点数units
        kernel_shapes = (input_units, self.output_units)
        self.kernel_real = self.add_weight(shape=kernel_shapes,
                                      initializer=self.kernel_initializer_real,
                                      name='kernel_real',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.kernel_image = self.add_weight(shape=kernel_shapes,
                                      initializer=self.kernel_initializer_image,
                                      name='kernel_image',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        # self.kernel_real = self.kernel[0]
        # self.kernel_image = self.kernel[1]
        if self.use_bias:
            self.bias_real = self.add_weight(shape=(self.output_units,),
                                             initializer=self.bias_initializer,
                                             name='bias_real',
                                             regularizer=self.bias_regularizer,
                                             constraint=self.bias_constraint)
            self.bias_image = self.add_weight(shape=(self.output_units,),
                                              initializer=self.bias_initializer,
                                              name='bias_image',
                                              regularizer=self.bias_regularizer,
                                              constraint=self.bias_constraint)
        else:
            self.bias_real = None
            self.bias_image = None

        # self.input_spec = InputSpec(min_ndim = 2, axes = {-1: input_units})
        self.built = True

    # 实际计算方法，传入inputs（不能被具体定义，必须是形式）
    def call(self, inputs, **kwargs):
        assert isinstance(inputs, list)
        input_real, input_image = inputs[0], inputs[1]
        input_shape = K.int_shape(input_real)

        # 实现复数计算方法
        output_real = K.dot(input_real, self.kernel_real) - K.dot(input_image, self.kernel_image)
        output_image = K.dot(input_image, self.kernel_real) + K.dot(input_real, self.kernel_image)

        if self.use_bias:
            output_real = K.bias_add(output_real, self.bias_real)
            output_image = K.bias_add(output_image, self.bias_image)
        if self.activation is not None:
            output_real = self.activation(output_real)
            output_image = self.activation(output_image)
        return output_real, output_image

    def compute_output_shape(self, input_shape):
        input_shapes = input_shape
        assert (input_shapes[0] == input_shapes[1])
        input_shape = input_shapes[0]
        assert len(input_shape) >= 2

        output_shape = list(input_shape)  # tuple 无法赋值
        output_shape[-1] = self.units     # 修改shape为output_shape
        output_shape = tuple(output_shape)   # shape一般为tuple类型
        return [output_shape] * 2            # 以列表形式返回

    def get_config(self):
        config = {
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': cinitializers.serialize(self.kernel_initializer),
            'bias_initializer': cinitializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint),
            'seed': self.seed,
        }
        base_config = super(ComplexDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))