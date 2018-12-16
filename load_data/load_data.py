import tensorflow as tf
import os
import scipy.io as scio
import numpy as np
from PIL import Image

'''
def load_image(filename):
    image_file = tf.read_file(filename)
    # Decode the image as a JPEG file, this will turn it into a Tensor
    image = tf.image.decode_jpeg(image_file)
    image = 255.0 * tf.image.convert_image_dtype(image, tf.float32)
    return image
'''


def load_image(filename):
    image = np.array(Image.open(filename), dtype=float)
    return image


def load_label(filename):
    data = scio.loadmat(filename)
    joints = data['joints']
    joints = np.transpose(joints)
    return joints


def joints_to_heatmap(joints):
    heatmap = np.zeros([64, 64, 14], dtype=float)
    for i in range(14):
        x = int(joints[i][0]/4)
        y = int(joints[i][1]/4)
        occlusion = joints[i][2] > 0.5
        if not occlusion:
            heatmap[x][y][i] = 1
            heatmap[x+1][y][i] = 1
            heatmap[x][y+1][i] = 1
            heatmap[x+1][y+1][i] = 1
    return heatmap


def load_data(image_path,  label_path,  batch_size):
    # 从指定路径中读取图片，得到图片的路径的list以及他们的id
    image_paths = []
    ids = []
    my_id = 0
    docs = sorted(os.listdir(image_path))
    for doc in docs:
        doc_dir = os.path.join(image_path, doc)
        if doc_dir.endswith('.jpg') or doc_dir.endswith('.jpeg'):
            image_paths.append(doc_dir)
            ids.append(my_id)
        my_id += 1

    # 将image_paths 和 ids 转换为tf可以处理的格式
    image_paths = tf.convert_to_tensor(image_paths, tf.string)
    ids = tf.convert_to_tensor(ids, tf.int32)

    # 读取label所在文件，得到image_labels这个list
    image_labels = load_label(label_path)

    # 建立Queue
    image_path, image_id, image_label = tf.train.slice_input_producer([image_paths, ids, image_labels], shuffle=True)

    # 读取图片，并进行解码
    image = tf.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)

    # 对图片进行裁剪和正则化（将数值[0,255]转化为[-1,1]）
    image = tf.image.resize_images(image, size=[256, 256])

    # 创建 batch
    x, y, z = tf.train.batch([image, image_id, image_label], batch_size=batch_size, num_threads=4, capacity=batch_size * 8)
    return x, y, z
