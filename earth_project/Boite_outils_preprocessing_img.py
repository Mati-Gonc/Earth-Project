import os

import cv2
import matplotlib.pyplot as plt
from google.cloud import storage


# Fonction permettant de décomposer une image en plusieurs secteurs
def reshape_split(image, kernel_size: tuple):

    img_height, img_width, channels = image.shape
    tile_height, tile_width = kernel_size



    tiled_array = image.reshape (img_height // tile_height,
                                 tile_height,
                                 img_width // tile_width,
                                 tile_width,
                                 channels)

    tiled_array = tiled_array.swapaxes(1, 2)

    return tiled_array


# Fonction permettant l'affichage recomposé en vignettes de l'image transformée par la fonction reshape_split
def show_reshape_split(image, nrows, ncols, figsize: tuple):
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    for j in range(nrows):
        for i in range(ncols):
            axs[j][i].imshow(image[j][i])

# Fonction permettant de télécharger un fichier depuis GoogleCloud
def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"

    # The ID of your GCS object
    # source_blob_name = "storage-object-name"

    # The path to which the file should be downloaded
    # destination_file_name = "local/path/to/file"

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)

    # Construct a client side representation of a blob.
    # Note `Bucket.blob` differs from `Bucket.get_blob` as it doesn't retrieve
    # any content from Google Cloud Storage. As we don't need additional data,
    # using `Bucket.blob` is preferred here.
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

    print(
        "Downloaded storage object {} from bucket {} to local file {}.".format(
            source_blob_name, bucket_name, destination_file_name
        )
    )


# Fonction permettant la création d'un nouveau bucket sur GoogleCloud
def create_bucket(bucket_name):
    """Creates a new bucket."""
    # bucket_name = "your-new-bucket-name"

    storage_client = storage.Client(project='earth-project-370013')

    bucket = storage_client.create_bucket(bucket_name, location="europe-west9")
    bucket.storage_class = "COLDLINE"

    print(f"Bucket {bucket.name} created")



# Fonction permettant d'uploader un fichier depuis GoogleCloud
def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"
    # The path to your file to upload
    # source_file_name = "local/path/to/file"
    # The ID of your GCS object
    # destination_blob_name = "storage-object-name"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print(
        f"File {source_file_name} uploaded to {destination_blob_name}."
    )





#slicing a picutre into 81 smaller picture then naming each picutre with a specific name - eg: 116_1_sat.jpg

folder_dir = "/data/train/"
folder_re_dir = "/data/sliced_img/"

def name_new_image_index(image_name, index):
    debut = image_name[:-8]
    fin = image_name[-8:]
    return f'{debut}_{index+1}{fin}'

def ninja_slicing_naming(folder_dir,folder_re_dir):

    for image in os.listdir(folder_dir) :
        if (image.endswith(".jpg")):
            image_sat = plt.imread(f'{folder_dir}{image}')
            r_image_sat = reshape_split(image_sat, (272,272))
            count=0
            for i in range(9):
                for j in range(9):
                    count+=1
                    cv2.imwrite(f'{folder_re_dir}{name_new_image_index(image, count)}', r_image_sat[i][j])

    return print('Burabō')


#same idea as presented above but with better style. #Roger_federer

def roger_slicing_naming(folder_dir,folder_re_dir):

    for image_name in os.listdir(folder_dir) :

        if 'mask'in image_name :
            pass

        else :
            image_sat = plt.imread(f'{folder_dir}/{image_name}')
            r_image_sat = reshape_split(image_sat, (272,272))

            for i, row in enumerate(r_image_sat):
                for j, image in enumerate(row):
                        filename = image_name.replace("_", f"_{9*i + j}_")
                        filename = image_name.replace("_", f"_{9*i + j}_")
                        cv2.imwrite(f'{folder_re_dir}/{filename}', image)

    return print('No more slicing bro')
