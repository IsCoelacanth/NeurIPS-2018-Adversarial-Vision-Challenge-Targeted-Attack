import tensorflow as tf
import os, csv
import resnetALP.model_lib as model_lib
from foolbox2.models.tensorflow import TensorFlowModel
from scipy.misc import imread, imsave
import PIL.Image
import numpy as np
from resnet18.resnet_model import Model as ResNetModel

def create_model():
    graph = tf.Graph()

    with graph.as_default():
        images = tf.placeholder(tf.float32, (None, 64, 64, 3))

        # preprocessing
        # _R_MEAN = 123.68
        # _G_MEAN = 116.78
        # _B_MEAN = 103.94
        # _CHANNEL_MEANS = [_R_MEAN, _G_MEAN, _B_MEAN]
        # features = images - tf.constant(_CHANNEL_MEANS)
        features = tf.multiply( tf.subtract( tf.divide(images, 255), 0.5), 2.0)
        model_fn_two_args = model_lib.get_model('resnet_v2_50', 200)
        logits = model_fn_two_args(features, is_training = False)

        # variables_to_restore = tf.contrib.framework.get_variables_to_restore()
        with tf.variable_scope('utilities'):
            saver = tf.train.Saver()
    return graph, saver, images, logits

#
# def create_model():
#     graph = tf.Graph()
#
#     with graph.as_default():
#         images = tf.placeholder(tf.float32, (None, 64, 64, 3))
#
#         # preprocessing
#         _R_MEAN = 123.68
#         _G_MEAN = 116.78
#         _B_MEAN = 103.94
#         _CHANNEL_MEANS = [_R_MEAN, _G_MEAN, _B_MEAN]
#         features = images - tf.constant(_CHANNEL_MEANS)
#
#         resnetmodel = ResNetModel(
#             resnet_size=18,
#             bottleneck=False,
#             num_classes=200,
#             num_filters=64,
#             kernel_size=3,
#             conv_stride=1,
#             first_pool_size=0,
#             first_pool_stride=2,
#             second_pool_size=7,
#             second_pool_stride=1,
#             block_sizes=[2, 2, 2, 2],
#             block_strides=[1, 2, 2, 2],
#             final_size=512,
#             version=2,
#             data_format=None)
#
#         logits = resnetmodel(features, False)
#         # You can add more models here trained on tiny imagenet
#         # https://github.com/pat-coady/tiny_imagenet/tree/master/src
#             # add more models here
#
#         with tf.variable_scope('utilities'):
#             saver = tf.train.Saver()
#
#     return graph, saver, images, logits


def create_fmodel():
    graph, saver, images, logits = create_model()
    sess = tf.Session(graph=graph)
    path = os.path.dirname(os.path.abspath(__file__))
    # path = os.path.join(path, 'resnet18', 'checkpoints', 'model')
    # saver.restore(sess, tf.train.latest_checkpoint(path))
    path = os.path.join(path, 'tiny_imagenet_alp05_2018_06_26.ckpt', 'tiny_imagenet_alp05_2018_06_26.ckpt')
    saver.restore(sess, path)

    with sess.as_default():
        fmodel = TensorFlowModel(images, logits, bounds=(0, 255))
    return fmodel

def read_images():
    data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "flower")
    with open(os.path.join(data_dir, "target_class.csv")) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            yield (row[0], np.array(PIL.Image.open(os.path.join(data_dir, row[1])).convert("RGB")), int(row[2]))

if __name__ == '__main__':
    # executable for debuggin and testing
    model = create_fmodel()
    for (file_name, image, label) in read_images():
        print(file_name, np.argmax(model.predictions(image)))

    # for (file_name, image, label) in read_images():
    #     logits = model.predictions(image)
    #     print(logits)
