## COMMON
import numpy as np
from foolbox2.criteria import TargetClass
## ADVERSARIAL ATTACK utilities
from adversarial_vision_challenge import load_model
from adversarial_vision_challenge import read_images
from adversarial_vision_challenge import store_adversarial
from adversarial_vision_challenge import attack_complete

from foolbox2.attacks.saliency import SaliencyMapAttack

def run_attack(model, image, target_class, pos_salience):
    criterion = TargetClass(target_class)
    attack = SaliencyMapAttack(model, criterion)
    original_label = np.argmax(model.predictions(image))
    return attack(image, original_label)

def main():
    forward_model = load_model()
    for (file_name, image, label) in read_images():
        # tf.logging.info('Checking image is np array: %s' % str(type(image) is np.ndarray))
        adversarial = run_attack(forward_model, image, label, None)
        store_adversarial(file_name, adversarial)
    # Announce that the attack is complete
    # NOTE: In the absence of this call, your submission will timeout
    # while being graded.
    # print("Attack is complete")
    attack_complete()

if __name__ == '__main__':
    main()
