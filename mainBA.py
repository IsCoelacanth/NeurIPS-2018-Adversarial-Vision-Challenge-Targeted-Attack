## COMMON
import numpy as np
from foolbox2.criteria import TargetClass
## ADVERSARIAL ATTACK utilities
from adversarial_vision_challenge import load_model
from adversarial_vision_challenge import read_images
from adversarial_vision_challenge import store_adversarial
from adversarial_vision_challenge import attack_complete
## For Boundary Attack
from tiny_imagenet_loader import TinyImageNetLoader
from foolbox2.attacks.boundary_attack3 import BoundaryAttack
##### FOR ITERATIVE GRADIENT ATTACK
from smiterative2 import SAIterativeAttack, RMSIterativeAttack, \
                        AdamIterativeAttack, AdagradIterativeAttack
from foolbox2.distances import MeanSquaredDistance
from fmodel2 import create_fmodel as create_fmodel_ALP
from fmodel5 import create_fmodel as create_fmodel_ALP1000
from foolbox2.models.wrappers2 import CompositeModel
from foolbox2.adversarial import Adversarial

def distance(X, Y):
    assert X.dtype == np.uint8
    assert Y.dtype == np.uint8
    X = X.astype(np.float64) / 255
    Y = Y.astype(np.float64) / 255
    return np.linalg.norm(X - Y)

def run_attack(loader, model, image, target_class):
    assert image.dtype == np.float32
    assert image.min() >= 0
    assert image.max() <= 255

    starting_point, calls, is_adv = loader.get_target_class_image(
        target_class, model)

    if not is_adv:
        print('could not find a starting point')
        return None

    criterion = TargetClass(target_class)
    original_label = model(image)
    # we can optimize the number of iterations
    iterations = (1000 - calls) // 7

    attack = BoundaryAttack(model, criterion)
    return attack(image, original_label, iterations=iterations,
                  max_directions=10, tune_batch_size=False,
                  starting_point=starting_point)

def run_attack2(model, image, target_class, pos_salience):
    criterion = TargetClass(target_class)
    # model == Composite model
    # Backward model = substitute model (resnet vgg alex) used to calculate gradients
    # Forward model = black-box model
    distance = MeanSquaredDistance
    attack = AdamIterativeAttack()
    # attack = foolbox.attacks.annealer(model, criterion)
    # prediction of our black box model on the original image
    original_label = np.argmax(model.predictions(image))
    adv = Adversarial(model, criterion, image, original_label, distance=distance)
    return attack(adv, pos_salience = pos_salience)

def main():
    loader = TinyImageNetLoader()

    forward_model = load_model()
    backward_model1 = create_fmodel_ALP()
    backward_model2 = create_fmodel_ALP1000()
    model = CompositeModel(
        forward_model=forward_model,
        backward_models=[backward_model1, backward_model2],
        weights = [0.5, 0.5])
    for (file_name, image, label) in read_images():
        adversarial = run_attack(loader, forward_model, image, label)
        if adversarial is None:
            adversarial = run_attack2(model, image, label, None)
        store_adversarial(file_name, adversarial)

    # Announce that the attack is complete
    # NOTE: In the absence of this call, your submission will timeout
    # while being graded.
    attack_complete()


if __name__ == '__main__':
    main()
