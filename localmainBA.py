import numpy as np
from foolbox2.criteria import TargetClass
from foolbox2.attacks.boundary_attack import BoundaryAttack
# from adversarial_vision_challenge import load_model
# from adversarial_vision_challenge import read_images
# from adversarial_vision_challenge import store_adversarial
# from adversarial_vision_challenge import attack_complete
from fmodel import create_fmodel as create_fmodel_18
from tiny_imagenet_loader import TinyImageNetLoader
import os, csv
from scipy.misc import imread, imsave
import PIL.Image
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

    iterations = (1000 - calls - 1) // 10 // 2

    attack = BoundaryAttack(model, criterion)
    return attack(image, original_label, iterations=iterations,
                  max_directions=10, tune_batch_size=False,
                  starting_point=starting_point)


def read_images():
    data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "flower")
    with open(os.path.join(data_dir, "target_class.csv")) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            yield (row[0], np.array(PIL.Image.open(os.path.join(data_dir, row[1])).convert("RGB")).astype(np.float32), int(row[2]))

def attack_complete():
    pass

def store_adversarial(file_name, adversarial):
    out_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output")
    imsave(os.path.join(out_dir, file_name + ".png"), adversarial, format="png")

def load_model():
    return create_fmodel_18()

def main():
    loader = TinyImageNetLoader()

    model = load_model()
    for (file_name, image, label) in read_images():
        adversarial = run_attack(loader, model, image, label)
        store_adversarial(file_name, adversarial)

    # Announce that the attack is complete
    # NOTE: In the absence of this call, your submission will timeout
    # while being graded.
    attack_complete()


if __name__ == '__main__':
    main()
