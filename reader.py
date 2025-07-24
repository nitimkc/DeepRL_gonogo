
from pathlib import Path
import time
import re
# import cv2
from torchvision.io import decode_image
import numpy as np
import matplotlib.pyplot as plt


class ImageReader(object):

    def __init__(self, root, fileids=None):

        self._fileids = fileids
        
    def _read_images(self, fileids):
        for path in fileids:
            # image = cv2.imread(path, 0)
            image = decode_image(path)
            yield image

    def process_image(self, fileids=None):
        """
        # implement Mobin's suggestion here
        """
        # return image
           
    def sizes(self, fileids=None):
        # every path and filesize
        for path in fileids:
            yield path.stat().st_size
    
    def filepath(self, fileids=None):
        # every path and filesize
        for path in fileids:
            yield path

    def labels(self, fileids=None):
        """
        extract label given name
        """
        for path in fileids:
            fname = path.name
            ids = re.findall(r'\d+', fname)
            label = 0 if ids[0] == ids[1] else 1
            yield label