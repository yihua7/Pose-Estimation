import tensorflow as tf
import model.networks as networks
import load_data.load_data as load_data
import time
import os


class Stacked_Hourglass():
    def __init__(self, block_number, layers, out_dim, point_num, lr):
        self.block_number = block_number
        self.layers = layers
        self.out_dim = out_dim
        self.point_num = point_num
        self.lr = lr
        self.input = tf.placeholder(tf.float32, shape=[None, 256, 256, 3], name='input_image')
        self.label = tf.placeholder(tf.float32, shape=[None, 64, 64, 14], name='input_label')

        self.mid = networks.set_conv(self.input, 6, 64, 2, 'compression')  # down sampling
        self.mid = networks.set_res(self.mid, 128, 'compression_res0')
        self.mid = tf.nn.max_pool(self.mid, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # down sampling
        self.mid = networks.set_res(self.mid, 128, 'compression_res1')
        self.mid = networks.set_res(self.mid, out_dim, 'compression_res2')

        hgout0 = networks.set_hourglass(input=self.mid, layers=layers, out_dim=out_dim, scope='hourglass0')
        hgout_conv1 = networks.set_conv(hgout0, 1, out_dim, 1, 'hgout0_conv0')
        hgout_conv1 = tf.contrib.layers.batch_norm(hgout_conv1, 0.9, epsilon=1e-5, activation_fn = tf.nn.relu, scope = 'hgout0_batch')
        hgout_conv2 = networks.set_conv(hgout_conv1, 1, out_dim, 1, 'hgout0_conv1')

        pred = networks.set_conv(hgout_conv1, 1, point_num, 1, 'pred0')
        heat_map = [pred]
        heat_map_reshape = networks.set_conv(pred, 1, out_dim, 1, 'reshape0')

        hgin1 = tf.add_n([self.mid, hgout_conv2, heat_map_reshape])
        hgin = [hgin1]

        for i in range(1, self.block_number):
            hgout0 = networks.set_hourglass(input=hgin[i-1], layers=layers, out_dim=out_dim, scope='hourglass'+str(i))
            hgout_conv1 = networks.set_conv(hgout0, 1, out_dim, 1, 'hgout'+str(i)+'_conv0')
            hgout_conv1 = tf.contrib.layers.batch_norm(hgout_conv1, 0.9, epsilon=1e-5, activation_fn=tf.nn.relu,
                                                       scope='hgout'+str(i)+'_batch')
            hgout_conv2 = networks.set_conv(hgout_conv1, 1, out_dim, 1, 'hgout'+str(i)+'_conv1')

            pred = networks.set_conv(hgout_conv1, 1, point_num, 1, 'pred'+str(i))
            heat_map.append(pred)
            heat_map_reshape = networks.set_conv(pred, 1, out_dim, 1, 'reshape'+str(i))

            hgin1 = tf.add_n([hgin[i-1], hgout_conv2, heat_map_reshape])
            hgin.append(hgin1)

        self.output = tf.sigmoid(tf.reduce_sum(heat_map, 0))
        self.loss = tf.losses.mean_squared_error(self.output, self.label)
#        self.loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.output, labels=self.label)
#        self.optimizer = tf.train.AdamOptimizer(lr).minimize(self.loss)
        self.optimizer = tf.train.RMSPropOptimizer(lr).minimize(self.loss)

        self.saver = tf.train.Saver(max_to_keep=2)

    def train(self, image_path, label_path, batch_size, maxepoch):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        image_list = sorted(os.listdir(image_path))
        joints = load_data.load_label(label_path)

        num_data = len(image_list)
        start_data = 1

        with tf.Session(config=config) as sess:
            tf.global_variables_initializer().run()
            for i in range(0, maxepoch):
                start = time.time()

                image = []
                label = []
                if start_data > num_data - batch_size - 2:
                    start_data = 1
                for j in range(start_data, start_data+batch_size):
                    next_image = load_data.load_image(image_path+('\\%04d' % j)+'.jpg')
                    image.append(next_image)
                    next_heatmap = load_data.joints_to_heatmap(joints[j])
                    label.append(next_heatmap)
                start_data += batch_size

                sess.run([self.optimizer], feed_dict={self.input: image, self.label: label})
                loss = sess.run(self.loss, feed_dict={self.input: image, self.label: label})
                print("Epoch: [%5d|total] loss:%.8f" % (i, loss))

                if i % 200 == 0 and i != 0:
                    self.saver.save(sess, 'parameters/hourglass_model', global_step=i)
                end = time.time()
                print('time: %fs' % (end - start))

    def test(self, image, label, maxepoch):
        with tf.Session() as sess:
            para_path = tf.train.latest_checkpoint('parameters/')
            self.saver.restore(sess, para_path)
            for i in range(0, maxepoch):
                out, loss = sess.run([tf.sigmoid(self.output), self.loss], feed_dict={self.input:image, self.label:label})
                print(out, loss)
