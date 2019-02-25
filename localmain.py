import numpy as np
from foolbox2.criteria import TargetClass
from foolbox2.models.wrappers import CompositeModel
from fmodel import create_fmodel
from foolbox2.attacks.iterative_projected_gradient import MomentumIterativeAttack
from foolbox2.distances import MeanSquaredDistance
from foolbox2.adversarial import Adversarial
# from adversarial_vision_challenge import load_model
# from adversarial_vision_challenge import read_images
# from adversarial_vision_challenge import store_adversarial
# from adversarial_vision_challenge import attack_complete
from iterative import SAIterativeAttack, RMSIterativeAttack, AdamIterativeAttack, AdagradIterativeAttack
import os, csv
from scipy.misc import imread, imsave
import PIL.Image

def run_attack(model, image, target_class):
    criterion = TargetClass(target_class)
    # model == Composite model
    # Backward model = substitute model (resnet vgg alex) used to calculate gradients
    # Forward model = black-box model
    # distance = MeanSquaredDistance
    attack = RMSIterativeAttack(model, criterion)
    # attack = foolbox.attacks.annealer(model, criterion)
    # prediction of our black box model on the original image
    original_label = np.argmax(model.predictions(image))
    # adv = Adversarial(model, criterion, image, original_label, distance=distance)
    # return attack(adv)
    return attack(image, model_path = None, label = original_label)

def main():
    # instantiate blackbox and substitute model
    forward_model = load_model()
    backward_model = create_fmodel()

    # instantiate differntiable composite model
    # (predictions from blackbox, gradients from substitute)
    model = CompositeModel(
        forward_model=forward_model,
        backward_model=backward_model)
    for (file_name, image, label) in read_images():
        adversarial = run_attack(model, image, label)
        store_adversarial(file_name, adversarial)
    # Announce that the attack is complete
    # NOTE: In the absence of this call, your submission will timeout
    # while being graded.
    attack_complete()

def read_images():
    data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "flower")
    with open(os.path.join(data_dir, "target_class.csv")) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            yield (row[0], np.array(PIL.Image.open(os.path.join(data_dir, row[1])).convert("RGB")), int(row[2]))

def attack_complete():
    pass

def store_adversarial(file_name, adversarial):
    out_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output")
    imsave(os.path.join(out_dir, file_name + ".png"), adversarial, format="png")

def load_model():
    return create_fmodel()

if __name__ == '__main__':
    main()
