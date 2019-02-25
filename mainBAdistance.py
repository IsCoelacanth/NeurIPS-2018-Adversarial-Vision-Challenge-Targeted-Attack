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
from foolbox2.attacks.boundary_attack2 import BoundaryAttack
##### FOR ITERATIVE GRADIENT ATTACK
from smiterative2 import AdamIterativeAttack
from foolbox2.distances import MeanSquaredDistance
from fmodel2 import create_fmodel as create_fmodel_ALP
from fmodel5 import create_fmodel as create_fmodel_ALP1000
from foolbox2.models.wrappers2 import CompositeModel
from foolbox2.adversarial import Adversarial

def distance(X, Y):
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
    iterations = (1000 - calls - 1) // 7

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
        weights = [0.3, 0.7])
    i = 0
    total_sum = 0.0
    prev_avg =0.0
    for (file_name, image, label) in read_images():
        is_boundary = True
        is_adam = True
        adversarial = run_attack(loader, forward_model, image, label)
        # Calculate both Adam and Boundary for first 5 images.
        if i < 5:
            adversarial_adam = run_attack2(model, image, label, None)
            if adversarial is not None:
                error1 = distance(adversarial, image)
            else:
                error1 = 100000.0
                is_boundary = False
            if adversarial_adam is not None:
                error_adam = distance(adversarial_adam, image)
            else:
                error_adam = 200000.0
                is_adam = False
            if is_adam and error1 - error_adam > 0.0:
                    adversarial = adversarial_adam
            if is_adam or is_boundary:
                i+=1
                total_sum += min(error1, error_adam)
        else:
            if adversarial is not None:
                error1 = distance(adversarial, image)
                prev_avg = total_sum/i
                i+=1
                total_sum += error1
            else:
                error1 = 100000.0
            if error1 > 25.0 or error1 > prev_avg or adversarial is None:
                adversarial_adam = run_attack2(model, image, label, None)
                if adversarial_adam is not None:
                    error_adam = distance(adversarial_adam, image)
                else:
                    error_adam = 200000.0
                if error1 - error_adam > 0.0:
                    adversarial = adversarial_adam
        store_adversarial(file_name, adversarial)

    # Announce that the attack is complete
    # NOTE: In the absence of this call, your submission will timeout
    # while being graded.
    attack_complete()


if __name__ == '__main__':
    main()
