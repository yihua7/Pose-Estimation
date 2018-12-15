import tensorflow as tf


def load_data():
    filename = '/data/1.jpg'
    image = load_image(filename)
    label = load_label(filename)
    return image, label


def load_image(filename):
    image_file = tf.read_file(filename)
    # Decode the image as a JPEG file, this will turn it into a Tensor
    image = tf.image.decode_jpeg(image_file)
    image = 255.0 * tf.image.convert_image_dtype(image, tf.float32)
    return image


def load_label(filename):
    return []