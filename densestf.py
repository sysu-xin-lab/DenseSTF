from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
# layerlist = [8, 12, 16, 6]  # 32466388


class Conv2DWeightNorm(tf.layers.Conv2D):
    def build(self, input_shape):
        self.wn_g = self.add_weight(
            name='wn_g',
            shape=(self.filters, ),
            dtype=self.dtype,
            initializer=tf.initializers.ones,
            trainable=True,
        )
        super(Conv2DWeightNorm, self).build(input_shape)
        square_sum = tf.reduce_sum(tf.square(self.kernel), [0, 1, 2], keepdims=False)
        inv_norm = tf.rsqrt(square_sum)
        self.kernel = self.kernel * (inv_norm * self.wn_g)


def conv2d_weight_norm(inputs,
                       filters,
                       kernel_size,
                       strides=(1, 1),
                       padding='valid',
                       data_format='channels_last',
                       dilation_rate=(1, 1),
                       activation=None,
                       use_bias=True,
                       kernel_initializer=None,
                       bias_initializer=tf.zeros_initializer(),
                       kernel_regularizer=None,
                       bias_regularizer=None,
                       activity_regularizer=None,
                       kernel_constraint=None,
                       bias_constraint=None,
                       trainable=True,
                       name=None,
                       reuse=None):
    layer = Conv2DWeightNorm(filters=filters,
                             kernel_size=kernel_size,
                             strides=strides,
                             padding=padding,
                             data_format=data_format,
                             dilation_rate=dilation_rate,
                             activation=activation,
                             use_bias=use_bias,
                             kernel_initializer=kernel_initializer,
                             bias_initializer=bias_initializer,
                             kernel_regularizer=kernel_regularizer,
                             bias_regularizer=bias_regularizer,
                             activity_regularizer=activity_regularizer,
                             kernel_constraint=kernel_constraint,
                             bias_constraint=bias_constraint,
                             trainable=trainable,
                             name=name,
                             dtype=inputs.dtype.base_dtype,
                             _reuse=reuse,
                             _scope=name)
    return layer.apply(inputs)


def avg_pool2d(x, pool_size=[2, 2], stride=2, padding='VALID'):
    return tf.layers.average_pooling2d(inputs=x,
                                       pool_size=pool_size,
                                       strides=stride,
                                       padding=padding)


def max_pool2d(x, pool_size=[2, 2], stride=2, padding='VALID'):
    return tf.layers.max_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)


def transition(net, scope='transition'):
    with tf.name_scope(scope):
        in_channel = int(net.shape[-1])
        net = conv2d_weight_norm(net,
                                 int(in_channel * 0.5),
                                 1,
                                 padding='same',
                                 name=scope + '_conv1x1')
        net = max_pool2d(net)
    return net


def bottleneck(net, filters, scope='block'):
    with tf.name_scope(scope):
        net = conv2d_weight_norm(net, filters * 6, 3, padding='same', name=scope + '_conv1')
        net = tf.nn.relu(net, name=scope + '_relu')
        net = conv2d_weight_norm(net, filters, 3, padding='same', name=scope + '_conv3')
        return net


def denseblock(net, layers, filters, scope='block'):
    with tf.name_scope(scope):
        layers_concat = list()
        layers_concat.append(net)
        x = bottleneck(net, filters, scope=scope + '_bottleneck_' + str(0))
        layers_concat.append(x)
        for i in range(layers - 1):
            x = tf.concat(layers_concat, axis=3)
            x = bottleneck(x, filters, scope=scope + '_bottleneck_' + str(i + 1))
            layers_concat.append(x)
        x = tf.concat(layers_concat, axis=3)
        return x


def densestf(net, layerlist, filters, num_output_channels=1):
   with tf.variable_scope('input'):
        layers_concat = list()
        x = conv2d_weight_norm(net, filters, 3, strides=2, padding='same', name='input_conv3x3')
        layers_concat.append(x)
        x = conv2d_weight_norm(net, filters, 7, strides=2, padding='same', name='input_conv7x7')
        layers_concat.append(x)
        net = tf.concat(layers_concat, axis=3)
    with tf.variable_scope('dense_blocks'):  # 32*32 #
        for i in range(len(layerlist) - 1):
            net = denseblock(net, layerlist[i], filters, scope='block_' + str(i))
            net = transition(net, scope='trans_' + str(i))
    with tf.variable_scope('final_block'):  # 4*4 #
        net = denseblock(net, layerlist[-1], filters, scope='final_block')
        net = tf.nn.relu(net, name='final_block_relu')

    with tf.variable_scope('output'):  # 4*4 #
        width = int(net.shape[1])
        height = int(net.shape[2])    
        net = conv2d_weight_norm(net,
                                 num_output_channels, [width, height],
                                 padding='valid',
                                 name='fc1')

    return net
