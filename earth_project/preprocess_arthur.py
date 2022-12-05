import tensorflow as tf
import pandas as pd
import numpy as np
import os


def load_image(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=3)
    return img


def get_corresponding_mask(path):
    path_mask = tf.strings.regex_replace(path, "sat.jpg", "mask.png")
    mask = load_image(path_mask)
    return mask


def load_data(path):
    img = load_image(path)
    try:
        mask = get_corresponding_mask(path)
    except FileNotFoundError:
        mask = None
    return img, mask

def encoder(img, mask=None):
    mask = mask / 255
    tensor = tf.tensordot(
        tf.cast(mask, tf.int32), tf.constant([[2**0], [2**1], [2**2]]), axes=1
    )

    return img, tensor

def resizer(
    img, mask=None, n_samples=9, img_height=2448, img_width=2448, img_channels=3
):

    tile_height = 272
    tile_width = 272

    # n_samples = img_height//tile_height

    img_pad = tf.image.pad_to_bounding_box(
        img, 0, 0, tile_height * n_samples, tile_width * n_samples
    )

    img = tf.reshape(
        img_pad, (n_samples, tile_height, n_samples, tile_width, img_channels)
    )
    img = tf.transpose(img, (0, 2, 1, 3, 4))
    img = tf.reshape(img, (-1, tile_height, tile_width, img_channels))

    if mask:
        mask_pad = tf.image.pad_to_bounding_box(
            mask, 0, 0, tile_width * n_samples, tile_height * n_samples
        )
        mask = tf.reshape(
            mask_pad, (n_samples, tile_height, n_samples, tile_width, img_channels)
        )
        mask = tf.transpose(mask, (0, 2, 1, 3, 4))
        mask = tf.reshape(mask, (-1, tile_height, tile_width, img_channels))
        return img,mask


    return img

def rescaler(img, mask):
    scaler = tf.keras.layers.Rescaling(1.0 / 255)
    return scaler(img), mask

def binary_mask(img, mask):
    shape = tf.shape(mask)
    np_fun = lambda x: np.where(x == 2, 1, 0)
    mask_bin = tf.numpy_function(func=np_fun, inp=[mask], Tout=tf.int64)
    mask_bin = tf.reshape(mask_bin, shape)
    # mask_bin = tf.squeeze(mask_bin)
    # mask_bin=tf.cast(tf.equal(mask,[3]),tf.uint8)
    # indexes = tf.where(tf.equal(mask, 3))
    # mask_bin = tf.gather_nd(mask, indexes)
    return img, mask_bin


def process_set(path_data="../raw_data/", set_partition=0.2, batch_size=8):

    path_metadata = os.path.join(path_data,'metadata.csv')
    df = pd.read_csv(path_metadata).replace(np.nan, "")
    df['sat_image_path_2'] =path_data+'/' + df['sat_image_path']
    max_idx_train = int(len(df[df.split == "train"]) * set_partition)
    df_train = df[df.split == "train"][:max_idx_train]
    df_test = df[df.split == "train"][max_idx_train:]

    ds_paths_train = tf.data.Dataset.from_tensor_slices(df_train.sat_image_path_2)
    ds_paths_test = tf.data.Dataset.from_tensor_slices(df_test.sat_image_path_2)


    # datasets mapping
    ds_binary_train = (
        ds_paths_train.map(load_data)
        .map(rescaler)
        .map(resizer)
        .map(encoder)
        .map(binary_mask)
        .unbatch()
    )
    ds_binary_test = (
        ds_paths_test.map(load_data)
        .map(rescaler)
        .map(resizer)
        .map(encoder)
        .map(binary_mask)
        .unbatch()
    )

    ds_train = ds_binary_train.batch(batch_size).prefetch(4)
    ds_test = ds_binary_test.batch(batch_size).prefetch(4)

    return ds_train, ds_test

def process_predict_img(img):
    #img = tf.io.decode_jpeg(img, channels=3)

    #img = load_image(path) au cas où
    mask=None
    img, mask = rescaler(img, mask)
    img = resizer(img, mask)
    #img, mask = encoder(img__, mask__)

    return img
