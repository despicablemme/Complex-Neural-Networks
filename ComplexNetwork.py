import numpy as np
from keras.models import Model, Sequential
from keras.layers import Input, Activation, concatenate
from complexlayers.conv import ComplexConv1D as C_conv1d
from complexlayers.dense import ComplexDense as C_dense
from complexlayers.normalization import ComplexBatchNormalization as C_bn


spectral = False


def complex_network():
    """
    复数网络双Input层，model.fit()中：
        x={},两个Input()层以字典传入
        y=[], 因为是一个层的两个输入，因此以列表传入
    """
    data_in1 = Input(shape=(1, 20), name='input_real')
    data_in2 = Input(shape=(1, 20), name='input_image')
    [conv11, conv12] = C_conv1d(filters=16, kernel_size=3, strides=1, padding='same', name='cconv1d1',
                                spectral_parametrization=spectral)([data_in1, data_in2])
    [dense11, dense12] = C_dense(units=8, name='cdense1')([conv11, conv12])
    [bn11, bn12] = C_bn(name='cbn1')([dense11, dense12])
    [act1, act2] = Activation('relu')([bn11, bn12])
    model = Model(inputs=[data_in1, data_in2], outputs=[act1, act2])
    return model


x1 = np.random.normal(size=(30, 1, 20))
x2 = np.random.normal(size=(30, 1, 20))
x3 = np.random.normal(size=(30, 1, 8))
x4 = np.random.normal(size=(30, 1, 8))
c_net = complex_network()
c_net.summary()
c_net.compile(optimizer='adam', loss='mse')
c_net.fit(x={'input_real': x1, 'input_image': x2}, y=[x3, x4], batch_size=5, epochs=100)
