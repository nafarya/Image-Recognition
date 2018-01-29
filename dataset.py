import pickle as pkl

import numpy as np
import tensorflow as tf
from skimage.io import imsave

#parse dataset as image from /data_set/cifar_10/data_batch_* and add it to tensorboard.

filename = './data_set/cifar_10/data_batch_1'

def convert_keys_to_string(dictionary):
    """Recursively converts dictionary keys to strings."""
    if not isinstance(dictionary, dict):
        return dictionary
    return dict((str(k), convert_keys_to_string(v))
        for k, v in dictionary.items())


PIXELS_DIR = "pixel_data"
LABEL_FILE = "labels.txt"

def convert(data):
    if isinstance(data, bytes):  return data.decode('ascii')
    if isinstance(data, dict):   return dict(map(convert, data.items()))
    if isinstance(data, tuple):  return map(convert, data)
    return data

def unpack_file(fname):
    with open(fname, "rb") as f:
        result = pkl.load(f, encoding='bytes')
    return convert_keys_to_string(result)


def save_as_image(img_flat):
    # consecutive 1024 entries store color channels of 32x32 image
    img_R = img_flat[0:1024].reshape((32, 32))
    img_G = img_flat[1024:2048].reshape((32, 32))
    img_B = img_flat[2048:3072].reshape((32, 32))
    img = np.dstack((img_R, img_G, img_B))
    #imsave("picture.png", img)
    image = tf.expand_dims(img, 0)
    summary_op = tf.summary.image("plot", image)
    with tf.Session() as sess:
        summary = sess.run(summary_op)
        writer = tf.summary.FileWriter('./tensorboard/images')
        writer.add_summary(summary)
        writer.close()


def start():
    labels = {}
    data = unpack_file(filename)

    for i in range(10000):
        img_flat = data["b'data'"][i]
        fname = data["b'filenames'"][i]
        label = data["b'labels'"][i]
        save_as_image(img_flat)
        labels[fname] = label

    with open(LABEL_FILE, "w") as f:
        for (fname, label) in labels.iteritems():
            f.write("{0} {1}\n".format(fname, label))

#start()
