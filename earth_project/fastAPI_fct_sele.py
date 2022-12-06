from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image
from pydantic import BaseModel
from image_to_API import image_to_dict, image_from_dict
import team_base_model

appselenium = FastAPI()

@appselenium.get("/")
def index():
    return {'ok': "jusqu'ici tout va bien "}

class Item(BaseModel):
    image:str
    size:int
    height:int
    width:int
    channel:int

@appselenium.post("/predict")
def predict(item:Item):
    #Appel à la base
    np_array_img = image_from_dict(dict(item))
    y_pred = team_base_model.make_pred(np_array_img)
    #print(type(np_array_img), 'wesh')
    img_json=image_to_dict(y_pred)
    return {"image":img_json}
