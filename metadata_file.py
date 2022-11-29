import numpy as np
import pandas as pd

def new_metadata_file():
    path = '../Earth-Project/raw_data/metadata.csv'
    df = pd.read_csv(path)

    df = df[df["split"]=="train"]

    image_id_new = []
    sat_image_path = []
    mask_image_path = []

    for image_id in df["image_id"]:
        for j in range(1,81):
            image_id_new.append(str(image_id)+"_"+str(j))

    for image_id in image_id_new:
        sat_image_path.append("train/"+image_id+"_sat.jpg")
        mask_image_path.append("train/"+image_id+"_mask.png")

    df_new = {
    "image_id" : image_id_new,
    "sat_image_path" : sat_image_path,
    "mask_image_path" : mask_image_path
    }
    df_new=pd.DataFrame(df_new)

    df_new.to_csv("../Earth-Project/raw_data/metadata_process.csv")


if __name__=="__main__" :
    new_metadata_file()
