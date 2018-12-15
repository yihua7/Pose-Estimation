import tensorflow as tf
import model.networks as networks
import time


class Stacked_Hourglass():
    def __init__(self, block_number, layers, out_dim, lr):
        self.block_number = block_number
        self.layers = layers
        self.out_dim = out_dim
        self.lr = lr
        self.input = tf.placeholder(tf.float32, shape=[], name='input_image')
        self.label = tf.placeholder(tf.float32)
        self.saver = tf.train.Saver(max_to_keep=2)

        inter0 = networks.set_hourglass(input=self.input, layers=layers, out_dim=out_dim, scope='hourglass0')
        self.inter = [inter0]
        for i in range(1, self.block_number):
            inter = networks.set_hourglass(input=self.inter[i], layers=layers, out_dim=out_dim, scope='hourglass'+str(i))
            self.inter.append(inter)
        self.output = self.inter[self.block_number-1]

        self.loss = []
        self.optimizer = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def train(self, image, label, maxepoch):
        config = tf.ConfigProto()
        config.gpu_options.allow_grouth = True
        with tf.Session(config=config) as sess:
            tf.global_variables_initializer().run()
            for i in range(0, maxepoch):
                start = time.time()
                sess.run([self.optimizer], feed_dict={self.input: image, self.label: label})
                self.loss = sess.run(self.loss, feed_dict={self.input: image, self.label: label})
                print("Epoch: [%5d|total] loss:%.8f" % (i, self.loss))

                if i % 200 == 0 or i == maxepoch:
                    self.saver.save(sess, 'parameters/hourglass_model', global_step=i)
                end = time.time()
                print('time: %fs' % (end - start))

    def test(self, image, label):
        pass

