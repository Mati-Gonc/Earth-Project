import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Conv2DTranspose, concatenate,  Dropout, MaxPooling2D, UpSampling2D


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
            size=(2, 2), data_format=None, interpolation="nearest", **kwargs
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



## Pour DL DATA Sur le notebook

def decode_img(img):
  # Convert the compressed string to a 3D uint8 tensor
  img = tf.io.decode_jpeg(img, channels=3)
  # Resize the image to the desired size
  return tf.image.resize(img, [272,272])

def process_path(file_path):
    img_sat = tf.io.read_file(file_path)
    img_sat = tf.io.decode_jpeg(img_sat,channels=3)
    img_sat = tf.cast(img_sat, tf.float32)


    #file_path_mask = "../"+file_path.replace("sat","mask")[36:]
    file_path_mask = tf.strings.regex_replace(file_path,"_sat.jpg","_mask.png")
    img_mask = tf.io.read_file(file_path_mask)
    img_mask = tf.io.decode_png(img_mask,channels=3)
    img_mask = tf.cast(img_mask, tf.float32)
    return img_sat, img_mask

def data_path():

    path = 'raw_data/metadata.csv'
    df = pd.read_csv(path)

    df["prep_path"] = "raw_data/"
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
