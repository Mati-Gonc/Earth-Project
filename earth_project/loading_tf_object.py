import numpy as np
import pandas as pd
import tensorflow as tf


def process_path(file_path):
    img_sat = tf.io.read_file(file_path)
    img_sat = tf.io.decode_jpeg(img_sat,channels=3)
    img_sat = tf.cast(img_sat, tf.float32)

    file_path_mask = tf.strings.regex_replace(file_path,"_sat.jpg","_mask.png")
    img_mask = tf.io.read_file(file_path_mask)
    img_mask = tf.io.decode_png(img_mask,channels=3)
    img_mask = tf.cast(img_mask, tf.float32)

    return img_sat, img_mask

def data_path():

    path = '../raw_data/metadata_process.csv'
    df = pd.read_csv(path)

    df["prep_path"] = "../raw_data/"
    df["full_path_sat"] =df["prep_path"]+df["sat_image_path"]
    df= df[df["split"]=="train"]

    list_image_path=df["full_path_sat"]
    size_sample = int(round(len(list_image_path)*0.8,0))
    size_sample_val = int(round((len(list_image_path)-size_sample)/2))

    list_image_path_train=list_image_path[0:size_sample]
    list_image_path_val=list_image_path[size_sample:(size_sample+size_sample_val)]
    list_image_path_test = list_image_path[size_sample+size_sample_val:]

    train_ds = tf.data.Dataset.from_tensor_slices(list_image_path_train).map(process_path)
    val_ds = tf.data.Dataset.from_tensor_slices(list_image_path_val).map(process_path)
    test_ds = tf.data.Dataset.from_tensor_slices(list_image_path_test).map(process_path)

    return train_ds, val_ds,test_ds
