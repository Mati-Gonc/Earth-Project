from Boite_outils_preprocessing_img import create_bucket, reshape_split, roger_slicing_naming, upload_blob
import os
from os import listdir
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from google.cloud import storage

folder_dir = "/Users/nicolaslaportefaber/code/elmatador9/Earth-Project/data/train/"
folder_re_dir = "/Users/nicolaslaportefaber/code/elmatador9/Earth-Project/data/123/"

def pipeline_preproc_order(folder_dir) :

    create_bucket('img_preproc_9_9')

    roger_slicing_naming(folder_dir=folder_dir, folder_re_dir=folder_re_dir)

    for image in os.listdir(folder_re_dir):
        upload_blob('img_preproc_9_9',f'{folder_re_dir}{image}', image)
