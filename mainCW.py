import numpy as np
from foolbox2.criteria import TargetClass
from foolbox2.models.wrappers import CompositeModel
# from fmodel3 import create_fmodel_combo
# from fmodel import create_fmodel as create_fmodel_18
# from fmodel2 import create_fmodel as create_fmodel_ALP
# from fmodel5 import create_fmodel as create_fmodel_ALP1000
# from foolbox2.attacks.iterative_projected_gradient import MomentumIterativeAttack
from foolbox2.attacks.carlini_wagner import CarliniWagnerL2Attack
from foolbox2.distances import MeanSquaredDistance
from foolbox2.adversarial import Adversarial
# from adversarial_vision_challenge import load_model
# from adversarial_vision_challenge import read_images
# from adversarial_vision_challenge import store_adversarial
# from adversarial_vision_challenge import attack_complete
# from smiterative2 import SAIterativeAttack, RMSIterativeAttack, \
#                         AdamIterativeAttack, AdagradIterativeAttack
import sys
# from smiterative2 import SAIterativeAttack, RMSIterativeAttack, AdamIterativeAttack, AdagradIterativeAttack
import os

def run_attack(model, image, target_class):
    criterion = TargetClass(target_class)
    # model == Composite model
    # Backward model = substitute model (resnet vgg alex) used to calculate gradients
    # Forward model = black-box model
    distance = MeanSquaredDistance
    attack = CarliniWagnerL2Attack()
    # attack = foolbox.attacks.annealer(model, criterion)
    # prediction of our black box model on the original image
    original_label = np.argmax(model.predictions(image))
    adv = Adversarial(model, criterion, image, original_label, distance=distance)
    return attack(adv)

def main():
    # tf.logging.set_verbosity(tf.logging.INFO)
    # instantiate blackbox and substitute model
    # instantiate blackbox and substitute model
    forward_model = load_model()
    # backward_model1 = create_fmodel_18()
    # backward_model2 = create_fmodel_ALP()
    # backward_model3 = create_fmodel_ALP1000()
    # print(backward_model1[0])
    # instantiate differntiable composite model
    # (predictions from blackbox, gradients from substitute)
    # model = CompositeModel(
    #     forward_model = forward_model,
    #     backward_model = backward_model2)
    model = forward_model
    for (file_name, image, label) in read_images():
        # pos_salience = find_salience(predictor, image)
        adversarial = run_attack(model, image, label)
        store_adversarial(file_name, adversarial)
    attack_complete()

if __name__ == '__main__':
    main()
