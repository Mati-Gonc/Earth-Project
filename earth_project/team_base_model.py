import tensorflow as tf
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Conv2DTranspose, concatenate
#,  Dropout, MaxPooling2D, UpSampling2D
#from tensorflow.keras.losses import BinaryFocalCrossentropy, binary_crossentropy, binary_focal_crossentropy
from preprocess_arthur import process_predict_img


def EncoderMiniBlock(inputs, n_filters=32, dropout_prob=0.3, max_pooling=True):
    conv = tf.keras.layers.Conv2D(n_filters,
                  3,  # filter size
                  activation='relu',
                  padding='same',
                  kernel_initializer='HeNormal')(inputs)
    conv = tf.keras.layers.Conv2D(n_filters,
                  3,  # filter size
                  activation='relu',
                  padding='same',
                  kernel_initializer='HeNormal')(conv)
    conv = BatchNormalization()(conv, training=False)
    if dropout_prob > 0:
        conv = tf.keras.layers.Dropout(dropout_prob)(conv)
    if max_pooling:
        next_layer = tf.keras.layers.MaxPooling2D(pool_size = (2,2))(conv)
    else:
        next_layer = conv
    skip_connection = conv
    return next_layer, skip_connection

def DecoderMiniBlock(prev_layer_input, skip_layer_input, n_filters=32, up_sampling='Con2DTranspose'):

    if up_sampling == 'Con2DTranspose':
        prev_layer_input = Conv2DTranspose(
                    n_filters,
                    (3,3),
                    strides=(2,2),
                    padding='same'
                    )(prev_layer_input)

    elif up_sampling == 'up_sampling' :
        prev_layer_input = tf.keras.layers.UpSampling2D(
            size=(2, 2), data_format=None, interpolation="nearest"#, **kwargs
            )(prev_layer_input)

    else : pass

    merge = concatenate([prev_layer_input, skip_layer_input], axis=3)
    conv = Conv2D(n_filters,
                 3,
                 activation='relu',
                 padding='same',
                 kernel_initializer='HeNormal')(merge)
    conv = Conv2D(n_filters,
                 3,
                 activation='relu',
                 padding='same',
                 kernel_initializer='HeNormal')(conv)
    return conv

def model_build():
    img_inputs = tf.keras.Input(shape=(272, 272, 3))

    # Encoding
    conv, skip_1 = EncoderMiniBlock(img_inputs, n_filters=32, max_pooling=True)
    conv, skip_2 = EncoderMiniBlock(conv, n_filters=64, max_pooling=True)
    conv, skip_3 = EncoderMiniBlock(conv, n_filters=128, max_pooling=True)
    conv, skip_4 = EncoderMiniBlock(conv, n_filters=256, max_pooling=True)
    conv, skip_5 = EncoderMiniBlock(conv, n_filters=512, max_pooling=False)

    #decoding
    conv = DecoderMiniBlock(conv, skip_5, n_filters=512, up_sampling=False)
    conv = DecoderMiniBlock(conv, skip_4, n_filters=256, up_sampling='up_sampling')
    conv = DecoderMiniBlock(conv, skip_3, n_filters=128, up_sampling='up_sampling')
    conv = DecoderMiniBlock(conv, skip_2, n_filters=64, up_sampling='up_sampling')
    conv = DecoderMiniBlock(conv, skip_1, n_filters=32, up_sampling='up_sampling')
    #output = DecoderMiniBlock(conv, img_inputs, n_filters=3, up_sampling=False)

    output = tf.keras.layers.Conv2D( 1, 1, activation='sigmoid', padding='same', kernel_initializer='HeNormal')(conv)

    model = tf.keras.Model(inputs=img_inputs, outputs=output)

    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='binary_focal_crossentropy', metrics='accuracy')

    return model

""" def model_fit(model, ds_train, epochs=4, model_name):
    model.fit(ds_train, epochs=epochs)
    return model """

""" def model_load(model, model_name):
    return model.load_weights(model_name) """

#'loaded_models/model_3layers/model_light'
#'loaded_models/model_full_layers/test_model'
#'loaded_models/model_full_data/model_full_data'


def make_pred(img, weights = 'loaded_models/model_full_data/model_full_data'):
    img = process_predict_img(img)
    model = model_build()
    model.load_weights(weights)
    y_pred = model.predict(img)
    print(y_pred.shape)
    y_pred=y_pred.reshape(-1, 9, 9, 272, 272, 1).swapaxes(2,3).reshape(-1,9*272,9*272,1)
    y_pred = np.where(y_pred > .259, [255,255,255], [0,0,0])
    return y_pred
