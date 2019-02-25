import tensorflow as tf
import os
import resnetALP.model_lib as model_lib
from foolbox2.models.tensorflow import TensorFlowModel

from resnet18.resnet_model import Model as ResNetModel
############ # DEBUG:
import os, csv
from scipy.misc import imread, imsave
import PIL.Image
import numpy as np

def create_model_resnetALP():
    graph_resnetALP = tf.Graph()
    with graph_resnetALP.as_default():
        images = tf.placeholder(tf.float32, (None, 64, 64, 3))
        # preprocessing
        # _R_MEAN = 123.68
        # _G_MEAN = 116.78
        # _B_MEAN = 103.94
        # _CHANNEL_MEANS = [_R_MEAN, _G_MEAN, _B_MEAN]
        # features = images - tf.constant(_CHANNEL_MEANS)
        features = tf.multiply( tf.subtract( tf.divide(images, 255), 0.5), 2.0)
        model_fn_two_args = model_lib.get_model('resnet_v2_50', 200)
        logits_resnetALP = model_fn_two_args(features, is_training = False)
        # variables_to_restore = tf.contrib.framework.get_variables_to_restore()
        with tf.variable_scope('utilities_ALP'):
            saver_resnetALP = tf.train.Saver()
    return graph_resnetALP, saver_resnetALP, images, logits_resnetALP

def create_model_resnet18():
    graph_resnet18 = tf.Graph()

    with graph_resnet18.as_default():
        images2 = tf.placeholder(tf.float32, (None, 64, 64, 3))
        # preprocessing
        _R_MEAN = 123.68
        _G_MEAN = 116.78
        _B_MEAN = 103.94
        _CHANNEL_MEANS = [_R_MEAN, _G_MEAN, _B_MEAN]
        features = images2 - tf.constant(_CHANNEL_MEANS)
        resnetmodel = ResNetModel(resnet_size=18,bottleneck=False,num_classes=200,
            num_filters=64,kernel_size=3,conv_stride=1,first_pool_size=0,first_pool_stride=2,
            second_pool_size=7,second_pool_stride=1,block_sizes=[2, 2, 2, 2],
            block_strides=[1, 2, 2, 2], final_size=512,version=2, data_format=None)
        logits_resnet18 = resnetmodel(features, False)
        with tf.variable_scope('utilities'):
            saver_resnet18 = tf.train.Saver()
    return graph_resnet18, saver_resnet18, images2, logits_resnet18


def create_fmodel_combo():
    graph_resnet18, saver_resnet18, images2, logits_resnet18 = create_model_resnet18()
    graph_resnetALP, saver_resnetALP, images, logits_resnetALP = create_model_resnetALP()
    sessResNet18 = tf.Session(graph=graph_resnet18)
    sessResNetALP = tf.Session(graph=graph_resnetALP)
    # path = os.path.dirname(os.path.abspath(__file__))

    # with sessResNetALP.as_default():
        # with graph_resnetALP.as_default():
            # tf.global_variables_initializer().run()
    path = os.path.dirname(os.path.abspath(__file__))
    path_resnetALP = os.path.join(path, 'tiny_imagenet_alp05_2018_06_26.ckpt', 'tiny_imagenet_alp05_2018_06_26.ckpt')
    saver_resnetALP.restore(sessResNetALP, path_resnetALP)

    with sessResNetALP.as_default():
        with graph_resnetALP.as_default():
            # logits1 = logits_resnetALP
            fmodel2 = TensorFlowModel(images, logits_resnetALP, bounds=(0, 255))
    # sessResNetALP.close()

    path_resnet18 = os.path.join(path, 'resnet18', 'checkpoints', 'model')
    saver_resnet18.restore(sessResNet18, tf.train.latest_checkpoint(path_resnet18))

    with sessResNet18.as_default():
        with graph_resnet18.as_default():
            # logits2 = logits_resnet18
            fmodel1 = TensorFlowModel(images2, logits_resnet18, bounds=(0, 255))
    sessResNet18.close()
    # sessResNetALP.close()
    # print(logits_resnetALP)
    # with sessResNet18.as_default():
    #     fmodel1 = TensorFlowModel(images, logits_resnet18, bounds=(0, 255))
    # sessResNet18.close()

    # with sessResNetALP.as_default():
    #     fmodel2 = TensorFlowModel(images, logits_resnetALP, bounds=(0, 255))
    # sessResNetALP.close()

        # with graph_resnet18.as_default():
    #         tf.global_variables_initializer().run()
    #         # path = os.path.dirname(os.path.abspath(__file__))
    #         path_resnet18 = os.path.join(path, 'resnet18', 'checkpoints', 'model')
    #         saver_resnet18.restore(sessResNet18, tf.train.latest_checkpoint(path_resnet18))
    # sessResNet18.close()
    # graph = tf.Graph()
    # with graph.as_default():
    #     logits1 = fmodel1.predictions(image)
    #     tf.reset_default_graph()
    #     logits = (logits1 + logits2)/2
    # sess = tf.Session(graph=graph)
    # with sess.as_default():
    #     fmodel = TensorFlowModel(images, logits, bounds=(0, 255))
    # sess.close()
    # fmodel1.predictions(image) + fmodel2.predictions(image)
    return fmodel1, fmodel2


def read_images():
    data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "flower")
    with open(os.path.join(data_dir, "target_class.csv")) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            yield (row[0], np.array(PIL.Image.open(os.path.join(data_dir, row[1])).convert("RGB")), int(row[2]))

if __name__ == '__main__':
    # executable for debuggin and testing
    fmodel1, fmodel2 = create_fmodel()
    for (file_name, image, label) in read_images():
        logits = model.predictions(image)
        print(logits)
