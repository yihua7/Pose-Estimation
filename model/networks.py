import tensorflow as tf


def set_conv(X, W_shape, out_dim, stride, scope=None):
    with tf.variable_scope(scope or 'conv', reuse=tf.AUTO_REUSE):
        X = tf.contrib.layers.batch_norm(X, 0.9, epsilon=1e-5, activation_fn=tf.nn.relu)
        W = tf.get_variable('W', [W_shape, W_shape, X.get_shape().as_list()[3], out_dim], trainable=True, initializer=tf.random_normal_initializer(mean=0, stddev=0.1))
        b = tf.get_variable('b', out_dim, trainable=True, initializer=tf.random_normal_initializer)
        out = tf.nn.bias_add(tf.nn.conv2d(X, W, strides=[1, stride, stride, 1], padding='SAME'), b)
        out = tf.nn.relu(out)
        return out


def set_res(input, out_dim, scope):
    with tf.variable_scope(scope or 'res_momdule', reuse=tf.AUTO_REUSE):
        skip = set_conv(X=input, W_shape=1, out_dim=out_dim, stride=1, scope=scope+'_skip_')

        res = set_conv(X=input, W_shape=1, out_dim=out_dim/2, stride=1, scope=scope+'_res0_')
        res = set_conv(X=res, W_shape=3, out_dim=out_dim/2, stride=1, scope=scope+'_res1_')
        res = set_conv(X=res, W_shape=1, out_dim=out_dim, stride=1, scope=scope+'_res2_')

        return tf.add(res, skip)


def set_hourglass(input, layers, out_dim, scope=None):
    with tf.variable_scope(scope or 'hourglass', reuse=tf.AUTO_REUSE):
        if layers == 0:
            output = set_res(input=input, out_dim=out_dim, scope='res_module0')
        else:
            conv_out = set_res(input=input, out_dim=out_dim, scope='res_module'+str(layers))
            res_out = tf.nn.max_pool(conv_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            res_out = set_hourglass(input=res_out, layers=layers-1, out_dim=out_dim, scope=scope)
            res_out = tf.image.resize_nearest_neighbor(res_out, tf.shape(res_out)[1:3]*2, 'up_sample')
            output = tf.add(conv_out, res_out)
    return output


