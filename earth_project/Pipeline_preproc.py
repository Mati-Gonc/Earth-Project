import os
from os import listdir

import cv2
import matplotlib.pyplot as plt
from google.cloud import storage
from PIL import Image

from Boite_outils_preprocessing_img import (create_bucket, reshape_split,
                                            roger_slicing_naming, upload_blob)

folder_dir = "/Users/thomasmissonnier/Downloads/archive/train"
folder_re_dir = "/Users/thomasmissonnier/Downloads/test_img_split"

def pipeline_preproc_order(folder_dir, folder_re_dir) :

    #create_bucket('img_preproc_9_9')

    roger_slicing_naming(folder_dir=folder_dir, folder_re_dir=folder_re_dir)

    #for image in os.listdir(folder_re_dir):
    #    upload_blob('img_preproc_9_9',f'{folder_re_dir}/{image}', image)
