import pandas  as pd
import numpy as np
import os
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

# Création du dataset metadata
df = pd.read_csv('data/metadata.csv')
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


# Fonction permettant de binariser une image de masque multiclasse vers une image de classe 0 ou 1. Remarque : on peut changer de target si on le souhaite (cf le dictionnaire des classes)
def binarize_img(img, target='forest_land'):
    encoder = np.array([1,2,4])
    land_classes_norm = {key : values/255 for key,values in land_classes.items()}
    land_classes_encod = {key : np.dot(values, encoder) for key, values in land_classes_norm.items()}
    img_norm = img/255
    img_encod = np.dot(img_norm,encoder)
    binarize = lambda x : 1 if x==land_classes_encod[target] else 0
    img_binarized = np.array([[binarize(pixel) for pixel in row] for row in img_encod])
    return img_binarized

# Fonction permettant d'uploader les masques issus d'un dataframe metadata vers un destination path
def upload_binary_mask(df, destination_path):
    count = 0
    for i in range(len(df)):
        img=cv2.imread(os.path.join('data', df['mask_path'][i]))
        img_binarized = binarize_img(img)
        cv2.imwrite(os.path.join(destination_path, name_bin(df['mask_path'][i])), img_binarized)
        count+=1
        print(f'{count}/{len(df)} uploaded')
        #plt.imshow(img)
        #plt.show()
        #plt.imshow(img_binarized, cmap='gray')
        #plt.show()


# bout de code qui permet de télécharger et afficher les éléments masqués
for element in os.listdir('data/binary_flat/'):
    img = cv2.imread(os.path.join('data/binary_flat/', element), 0)
    X_bin.append(img)
    #plt.imshow(img, cmap='gray')
    #plt.show()
