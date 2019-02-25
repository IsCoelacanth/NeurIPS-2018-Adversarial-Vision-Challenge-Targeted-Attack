import os
import numpy as np
from PIL import Image


class TinyImageNetLoader(object):
    """Loader for images from the the Tiny ImageNet validation
    set for images of a specific class."""
    def __init__(self):
        with open(self.path('wnids.txt')) as f:
            wnids = f.readlines()
            assert len(wnids) == 200
            wnids = [x.strip() for x in wnids]
            self.wnids = wnids

        with open(self.path('val/val_annotations.txt')) as f:
            labels = f.readlines()
            assert len(labels) == 10000
            labels = [x.split('\t')[:2] for x in labels]
            images = {}
            for image, wnid in labels:
                assert wnid in self.wnids
                assert image.endswith('.JPEG')
                images.setdefault(wnid, []).append(image)

            assert len(images) == len(wnids)
            for wnid in images:
                assert len(images[wnid]) == 50
            self.images = images

    def path(self, *path):
        return os.path.join('tiny-imagenet-200', *path)

    def load_val_image(self, filename):
        path = self.path('val/images', filename)
        image = Image.open(path)
        image = np.asarray(image)
        if image.shape != (64, 64, 3):
            # e.g. grayscale
            return None
        assert image.dtype == np.uint8
        image = image.astype(np.float32)
        assert image.shape == (64, 64, 3)
        return image

    def get_target_class_image(self, label, model):
        """Loads images from the the Tiny ImageNet validation
        set until it finds one that's classified as the given
        class."""
        wnid = self.wnids[label]
        files = self.images[wnid]
        assert len(files) == 50
        for i, filename in enumerate(files):
            image = self.load_val_image(filename)
            if image is None:
                print('ignoring invalid image (e.g. grayscale)')
                continue
            if model(image) == label:
                # print('found an image from the target class')
                return image, i + 1, True
            else:
                # print('ignoring validation set image')
                continue

        print('could not find an image from the target class')
        # but we still return the last image,
        # the interpolation could in theory still work
        return image, len(files), False
