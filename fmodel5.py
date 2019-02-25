import tensorflow as tf
import os, csv
import resnetALP.model_lib as model_lib
from foolbox2.models.tensorflow import TensorFlowModel
from scipy.misc import imread, imsave
import PIL.Image
import numpy as np
from resnet18.resnet_model import Model as ResNetModel
import os, csv
from scipy.misc import imread, imsave
import PIL.Image

def mapping():
    return {515: 152, 7: 59, 8: 33, 10: 32, 11: 34, 12: 53, 525: 110, 527: 128, 528: 156, 531: 160, 532: 118, 533: 76, 535: 157, 24: 58, 539: 85, 542: 138, 543: 93, 545: 96, 547: 60, 548: 87, 559: 148, 52: 52, 566: 137, 570: 140, 59: 26, 61: 36, 608: 10, 578: 142, 585: 123, 354: 126, 591: 84, 594: 164, 83: 55, 598: 119, 600: 153, 601: 7, 603: 8, 607: 9, 96: 56, 611: 11, 101: 48, 617: 19, 618: 20, 108: 50, 786: 162, 622: 37, 629: 38, 630: 39, 631: 40, 121: 54, 634: 42, 635: 43, 125: 27, 639: 44, 643: 45, 645: 46, 647: 14, 649: 15, 653: 16, 654: 17, 655: 18, 658: 47, 147: 49, 151: 30, 668: 141, 671: 105, 672: 175, 675: 161, 165: 51, 678: 166, 680: 155, 681: 158, 193: 24, 683: 67, 173: 25, 174: 31, 176: 28, 699: 125, 190: 35, 705: 66, 707: 83, 708: 94, 289: 165, 712: 103, 713: 92, 716: 79, 718: 65, 720: 97, 722: 130, 211: 29, 213: 13, 633: 41, 729: 147, 975: 182, 220: 169, 733: 72, 734: 183, 735: 185, 962: 146, 738: 184, 806: 191, 746: 186, 236: 109, 238: 116, 751: 145, 754: 178, 757: 106, 762: 90, 256: 104, 770: 113, 777: 74, 266: 71, 727: 168, 268: 95, 270: 117, 274: 154, 275: 108, 788: 136, 790: 170, 795: 174, 284: 122, 285: 134, 799: 100, 801: 91, 294: 112, 813: 179, 814: 114, 817: 78, 819: 173, 820: 81, 310: 143, 137: 57, 313: 98, 827: 89, 830: 193, 319: 187, 320: 188, 323: 189, 836: 86, 326: 190, 839: 77, 843: 129, 845: 63, 846: 139, 847: 64, 851: 80, 855: 135, 866: 120, 315: 99, 359: 196, 361: 195, 876: 144, 365: 197, 366: 198, 367: 199, 880: 121, 677: 62, 882: 167, 884: 70, 374: 115, 887: 82, 378: 132, 896: 61, 901: 131, 905: 68, 908: 69, 923: 133, 925: 151, 930: 177, 420: 12, 423: 21, 936: 172, 327: 200, 945: 159, 947: 194, 948: 192, 951: 176, 440: 22, 441: 23, 450: 1, 968: 181, 971: 107, 974: 180, 333: 127, 595: 124, 980: 111, 985: 75, 986: 150, 987: 88, 476: 5, 989: 163, 991: 73, 995: 102, 486: 6, 1000: 101, 494: 2, 499: 3, 501: 4, 507: 149, 682: 171}

def create_model():
    graph = tf.Graph()
    with graph.as_default():
        images = tf.placeholder(tf.float32, (None, 64, 64, 3))
        features = tf.multiply( tf.subtract( tf.divide(images, 255), 0.5), 2.0)
        model_fn_two_args = model_lib.get_model('resnet_v2_50', 1001)
        logits = model_fn_two_args(features, is_training = False)

        # variables_to_restore = tf.contrib.framework.get_variables_to_restore()
        with tf.variable_scope('utilities'):
            saver = tf.train.Saver()

    return graph, saver, images, logits

def create_fmodel():
    graph, saver, images, logits = create_model()
    sess = tf.Session(graph=graph)
    path = os.path.dirname(os.path.abspath(__file__))
    # path = os.path.join(path, 'resnet18', 'checkpoints', 'model')
    # saver.restore(sess, tf.train.latest_checkpoint(path))
    path = os.path.join(path, 'imagenet64_alp025_2018_06_26.ckpt', 'imagenet64_alp025_2018_06_26.ckpt')
    saver.restore(sess, path)

    with sess.as_default():
        #logits_np_array = logits.eval()
        # new_logits = np.zeros([200,1])
        # dictionary = mapping()
        # for imagenet_index in dictionary:
        #     tiny_imagenet_index = dictionary[imagenet_index]
        #     new_logits[tiny_imagenet_index] = logits_np_array[imagenet_index]
        # new_logits_as_tensor = tf.convert_to_tensor(new_logits, np.float32)
        # fmodel = TensorFlowModel(images, new_logits_as_tensor, bounds=(0, 255))
        dictionary = mapping()
        imagenet_indices = list(dictionary.keys())
        unordered_tinyimagenet_logits = tf.transpose(tf.nn.embedding_lookup(tf.transpose(logits),imagenet_indices))
        tiny_imagenet_indices = []
        for k in dictionary:
            tiny_imagenet_indices.append(dictionary[k])
        ordered_tinyimagenet_logits = tf.transpose(tf.nn.embedding_lookup(tf.transpose(unordered_tinyimagenet_logits),imagenet_indices))
        fmodel = TensorFlowModel(images, ordered_tinyimagenet_logits, bounds=(0, 255))
    return fmodel
def read_images():
    data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "flower")
    with open(os.path.join(data_dir, "target_class.csv")) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            yield (row[0], np.array(PIL.Image.open(os.path.join(data_dir, row[1])).convert("RGB")), int(row[2]))

if __name__ == '__main__':
    model = create_fmodel()
    for (file_name, image, label) in read_images():
        model.predictions(image)
