import tensorflow as tf


def set_conv(X, W_shape, b_shape, stride, scope=None):
    with tf.variable_scope(scope or 'conv', reuse=tf.AUTO_REUSE):
        W = tf.get_variable('W', [1, W_shape, W_shape, 1], trainable=True, initializer=tf.random_normal_initializer(mean=0, stddev=0.1))
        b = tf.get_variable('b', b_shape, trainable=True, initializer=tf.random_normal_initializer)
        out = tf.nn.conv2d(X, W, strides=[1, stride, stride, 1], padding='SAME') + b
        out = tf.nn.relu(out)
        return out


def set_res(input, layers, w_dim, scope=None):
    with tf.variable_scope(scope or 'res_block', reuse=tf.AUTO_REUSE):
        if layers == 0:
            conv_out = set_conv(X=input, W_shape=w_dim[layers], b_shape=input.get_shape(), stride=1, scope='res0_0')
            conv_out = set_conv(X=conv_out, W_shape=w_dim[layers], b_shape=input.get_shape(), stride=1, scope='res0_1')
            output = set_conv(X=conv_out, W_shape=w_dim[layers], b_shape=input.get_shape(), stride=1, scope='res0_2')
        else:
            conv_out = set_conv(X=input, W_shape=w_dim[layers], b_shape=input.get_shape(), stride=1, scope='res'+str(layers))
            res_out = tf.nn.pool(conv_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            res_out = set_res(res_out, layers-1, w_dim)
            res_out = up_sample(res_out)
            output = conv_out + res_out
    return output


def up_sample(input):
    return input
    pass
