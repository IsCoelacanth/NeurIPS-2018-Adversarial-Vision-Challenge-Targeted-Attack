import numpy as np
from foolbox2.criteria import TargetClass
from foolbox2.models.wrappers import CompositeModel
from fmodel2 import create_fmodel
from foolbox2.distances import MeanSquaredDistance
from foolbox2.adversarial import Adversarial
from adversarial_vision_challenge import load_model
from adversarial_vision_challenge import read_images
from adversarial_vision_challenge import store_adversarial
from adversarial_vision_challenge import attack_complete
from smiterative2 import SAIterativeAttack, RMSIterativeAttack, \
                        AdamIterativeAttack, AdagradIterativeAttack
import os
# For Saliency Maps
import sys
from contextlib import contextmanager
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import resnet_v1
import tensorpack as tp
import tensorpack.utils.viz as viz
from skimage.transform import resize
IMAGE_SIZE = 224

@contextmanager
def guided_relu():
    """
    Returns:
        A context where the gradient of :meth:`tf.nn.relu` is replaced by
        guided back-propagation, as described in the paper:
        `Striving for Simplicity: The All Convolutional Net
        <https://arxiv.org/abs/1412.6806>`_
    """
    from tensorflow.python.ops import gen_nn_ops   # noqa

    @tf.RegisterGradient("GuidedReLU")
    def GuidedReluGrad(op, grad):
        return tf.where(0. < grad,
                        gen_nn_ops.relu_grad(grad, op.outputs[0]),
                        tf.zeros(grad.get_shape()))

    g = tf.get_default_graph()
    with g.gradient_override_map({'Relu': 'GuidedReLU'}):
        yield


def saliency_map(output, input, name="saliency_map"):
    """
    Produce a saliency map as described in the paper:
    `Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps
    <https://arxiv.org/abs/1312.6034>`_.
    The saliency map is the gradient of the max element in output w.r.t input.
    Returns:
        tf.Tensor: the saliency map. Has the same shape as input.
    """
    max_outp = tf.reduce_max(output, 1)
    saliency_op = tf.gradients(max_outp, input)[:][0]
    return tf.identity(saliency_op, name=name)

class SaliencyModel(tp.ModelDescBase):
    def inputs(self):
        return [tf.placeholder(tf.float32, (IMAGE_SIZE, IMAGE_SIZE, 3), 'image')]

    def build_graph(self, orig_image):
        mean = tf.get_variable('resnet_v1_50/mean_rgb', shape=[3])
        with guided_relu():
            with slim.arg_scope(resnet_v1.resnet_arg_scope()):
                image = tf.expand_dims(orig_image - mean, 0)
                logits, _ = resnet_v1.resnet_v1_50(image, 1000)
            saliency_map(logits, orig_image, name="saliency")

def find_salience(predictor, im):
    # resnet expect RGB inputs of 224x224x3
    im = resize(im, (IMAGE_SIZE, IMAGE_SIZE))
    im = im.astype(np.float32)[:, :, ::-1]
    # print(type(im))
    saliency_images = predictor(im)[0]
    # print(saliency_images)
    # print(type(saliency_images))
    pos_saliency = np.maximum(0, saliency_images)
    resized_pos_saliency = resize(pos_saliency, (64,64))
    # print(resized_pos_saliency.shape)
    return resized_pos_saliency

def run_attack(model, image, target_class, pos_salience):
    criterion = TargetClass(target_class)
    # model == Composite model
    # Backward model = substitute model (resnet vgg alex) used to calculate gradients
    # Forward model = black-box model
    distance = MeanSquaredDistance
    attack = SAIterativeAttack()
    # attack = foolbox.attacks.annealer(model, criterion)
    # prediction of our black box model on the original image
    original_label = np.argmax(model.predictions(image))
    adv = Adversarial(model, criterion, image, original_label, distance=distance)
    return attack(adv, pos_salience = pos_salience)

def main():
    # instantiate blackbox and substitute model
    forward_model = load_model()
    backward_model = create_fmodel()

    # instantiate differntiable composite model
    # (predictions from blackbox, gradients from substitute)
    model = CompositeModel(
        forward_model=forward_model,
        backward_model=backward_model)
    predictor = tp.OfflinePredictor(tp.PredictConfig(
        model=SaliencyModel(),
        session_init=tp.get_model_loader("resnet_v1_50.ckpt"),
        input_names=['image'],
        output_names=['saliency']))
    for (file_name, image, label) in read_images():
        pos_salience = find_salience(predictor, image)
        adversarial = run_attack(model, image, label, pos_salience)
        store_adversarial(file_name, adversarial)
    attack_complete()

if __name__ == '__main__':
    main()
