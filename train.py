import tensorflow as tf
import numpy as np
from load_data.load_data import load_data
from model.hourglass_net import Stacked_Hourglass

block_number = 8
layers = 3
lr = 0.0001
out_dim = 256
point_num = 14
maxepoch = 10001

image_path = 'D:\\CS\\机器学习大作业\\Pose-Detection\\data_set\\images_padding'
label_path = 'D:\\CS\\机器学习大作业\\Pose-Detection\\data_set\\joints\\joints.mat'
batch_size = 1



'''
with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord)
    i = 0
    try:
        while not coord.should_stop():
            imagee, idd, labell = sess.run([image, id, label])
            i += 1
            for j in range(5):
                print(imagee[j], labell[j])
    except tf.error.OutOfRangeError:
        print("done")
    finally:
        coord.request_stop()
    coord.join(threads)
'''
model = Stacked_Hourglass(block_number=block_number, layers=layers, out_dim=out_dim, point_num=point_num, lr=lr)
model.train(image_path, label_path, batch_size, maxepoch)
