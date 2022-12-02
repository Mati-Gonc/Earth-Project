import os
from os import listdir

import cv2
import numpy as np
import pandas as pd
from google.cloud import storage
from PIL import Image

import Boite_outils_preprocessing_img
import loading_tf_object
import masking
import metadata_file
import preprocess_arthur

destination_path = '../raw_data/train'
folder_dir = "/data/train/"
folder_re_dir = "/data/sliced_img/"

# Création du dataset metadata
df = pd.read_csv('../raw_data/metadata.csv')
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

    #masking.upload_binary_mask(df, destination_path)

    #print('masking done')

    #sliced_img_path = '../raw_data/train'
    #sliced_img_dir = "sliced_img"
    #sliced_img_fullpath = '../raw_data/sliced_img'

    #os.mkdir(os.path.join("Earth-Project","raw_data",sliced_img_dir))

    #Boite_outils_preprocessing_img.roger_slicing_naming(sliced_img_path, sliced_img_fullpath)

    #print('slicing done bitch')

    #metadata_file.new_metadata_file()

    #print('nouveau fichier metadata cree')

    #path_to_data_process = '../raw_data/metadata_process.csv'
    #train_ds, val_ds, test_ds = loading_tf_object.data_path(path_to_data_process)

    path_data = "../raw_data/"
    set_partition= 0.8

    ds_train, ds_test = preprocess_arthur.process_set(path_data, set_partition)
