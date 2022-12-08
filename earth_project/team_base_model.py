import tensorflow as tf
import pandas as pd
import numpy as np
import os
from tensorflow.keras.layers import Conv2D, BatchNormalization, Conv2DTranspose, concatenate
from keras.applications.vgg16 import VGG16
#,  Dropout, MaxPooling2D, UpSampling2D
#from tensorflow.keras.losses import BinaryFocalCrossentropy, binary_crossentropy, binary_focal_crossentropy
from earth_project.preprocess_arthur import process_predict_img

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

def DecoderMiniBlock(prev_layer_input, skip_layer_input, n_filters=32, up_sampling='up_sampling'):

    if up_sampling == 'Con2DTranspose':
        prev_layer_input = Conv2DTranspose(
                    n_filters,
                    (3,3),
                    strides=(2,2),
                    padding='same'
                    )(prev_layer_input)

    elif up_sampling == 'up_sampling' :
        prev_layer_input = tf.keras.layers.UpSampling2D(
            size=(2, 2), data_format=None, interpolation="nearest")(prev_layer_input)

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
#'/home/mati/code/Mati-Gonc/Earth-Project/earth_project/loaded_models/vgg_train_1/weights_vgg_train_1'


def unet_vgg_build():
    vggmodel = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    layers_name = [layer.name for layer in vggmodel.layers[1:]]
    skip_layers = {layer_name : vggmodel.get_layer(layer_name).output for layer_name in layers_name}

    vggmodel.trainable = False

    conv = vggmodel.output

    #conv =  tf.keras.layers.MaxPooling2D(pool_size = (2,2))(conv)
    #decoding
    conv = DecoderMiniBlock(conv, skip_layers['block5_conv3'], n_filters=512, up_sampling='up_sampling')
    conv = DecoderMiniBlock(conv, skip_layers['block4_conv3'], n_filters=256, up_sampling='up_sampling')
    conv = DecoderMiniBlock(conv, skip_layers['block3_conv3'], n_filters=128, up_sampling='up_sampling')
    conv = DecoderMiniBlock(conv, skip_layers['block2_conv2'], n_filters=64, up_sampling='up_sampling')
    conv = DecoderMiniBlock(conv, skip_layers['block1_conv2'], n_filters=32, up_sampling='up_sampling')
    #output = DecoderMiniBlock(conv, img_inputs, n_filters=3, up_sampling=False)

    output = tf.keras.layers.Conv2D( 1, 1, activation='sigmoid', padding='same', kernel_initializer='HeNormal')(conv)

    unet_vgg = tf.keras.Model(inputs=vggmodel.input, outputs=output)

    return unet_vgg

def unet_vgg_multi_build(nb_classes=4):
    #Encoding
    vggmodel = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    layers_name = [layer.name for layer in vggmodel.layers[1:]]
    skip_layers = {layer_name : vggmodel.get_layer(layer_name).output for layer_name in layers_name}

    vggmodel.trainable = False

    conv = vggmodel.output

    #decoding
    conv = DecoderMiniBlock(conv, skip_layers['block5_conv3'], n_filters=512, up_sampling='up_sampling')
    conv = DecoderMiniBlock(conv, skip_layers['block4_conv3'], n_filters=256, up_sampling='up_sampling')
    conv = DecoderMiniBlock(conv, skip_layers['block3_conv3'], n_filters=128, up_sampling='up_sampling')
    conv = DecoderMiniBlock(conv, skip_layers['block2_conv2'], n_filters=64, up_sampling='up_sampling')
    conv = DecoderMiniBlock(conv, skip_layers['block1_conv2'], n_filters=32, up_sampling='up_sampling')

    output = tf.keras.layers.Conv2D(nb_classes, 1, activation='softmax', padding='same', kernel_initializer='HeNormal')(conv)

    unet_vgg = tf.keras.Model(inputs=vggmodel.input, outputs=output)

    return unet_vgg


def make_pred(img, weights = os.path.join(os.environ['HOME'],'code/Mati-Gonc/Earth-Project/earth_project/loaded_models/vgg_train_1/weights_vgg_train_1')):
    img = process_predict_img(img)
    model = model_build()
    model.load_weights(weights)
    y_pred = model.predict(img)
    print(y_pred.shape)
    y_pred=y_pred.reshape(-1, 9, 9, 272, 272, 1).swapaxes(2,3).reshape(-1,9*272,9*272,1)
    y_pred = np.where(y_pred > .3, [255,255,255], [0,0,0])
    return y_pred

def make_pred_VGG(img, threshold=0.3, weights = os.path.join(os.environ['HOME'],'code/Mati-Gonc/Earth-Project/earth_project/loaded_models/vgg_train_1/weights_vgg_train_1')):
    img = process_predict_img(img)
    model = unet_vgg_build()
    model.load_weights(weights)
    y_pred = model.predict(img)
    print(y_pred.shape)
    y_pred=y_pred.reshape(-1, 11, 11, 224, 224, 1).swapaxes(2,3).reshape(-1,11*224,11*224,1)
    y_pred = np.where(y_pred > threshold, [255,255,255], [0,0,0])
    return y_pred


def make_pred_VGG_multi(img, nb_classes=4, weights=None):
    img = process_predict_img(img)
    model = unet_vgg_multi_build(nb_classes)
    model.load_weights(weights)
    y_pred = model.predict(img)
    print(y_pred.shape)

    y_pred=y_pred.reshape(-1, 11, 11, 224, 224, 4).swapaxes(2,3).reshape(-1,11*224,11*224,4)
    y_pred = np.argmax(y_pred, -1)
    return y_pred
