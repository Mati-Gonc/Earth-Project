import os
from os import listdir

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from google.cloud import storage
from PIL import Image

import masking
import Boite_outils_preprocessing_img
import metadata_file
import loading_tf_object

destination_path = 'raw_data/binary'
folder_dir = "/data/train/"
folder_re_dir = "/data/sliced_img/"

# Création du dataset metadata
df = pd.read_csv('raw_data/metadata.csv')
df = df[df['split'] == 'train']

# Dictionnaire des classes du masque d'origine
land_classes = {
    'urban_land': np.array([0,255,255]),
    'agriculture_land' : np.array([255,255,0]),
    'rangeland' : np.array([255, 0, 255]),
    'forest_land' : np.array([0, 255, 0]),
    'water' : np.array([0, 0, 255]),
    'barren_land' : np.array([255, 255, 255]),
    'unkown' : np.array([0, 0, 0])
}

if __name__ == '__main__':

    # /!\ CREER UN MASK DATA A METTRE EN DEST

    masking.upload_binary_mask(df, destination_path)

    # /!\ CREER UN DOSSIER SLICING_IMG

    Boite_outils_preprocessing_img.roger_slicing_naming(folder_dir, folder_re_dir)

    # /!\ RAJOUTER FONCTION POUR SLICING MASK /!\

    # /!\ METTRE UNE FONCTION QUI MET TOUTES LES IMAGES DANS PROCESS_DATA FOLDER

    metadata_file.new_metadata_file()

    train_ds, val_ds, test_ds = loading_tf_object.data_path()
