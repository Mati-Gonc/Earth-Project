from sys import displayhook
import streamlit as st

import streamlit as st
import numpy as np
from PIL import Image
import requests as rq
import json
from image_to_API import image_to_dict, image_from_dict
from geoportail import launch_search


'''
# Badass forest calculator 🔎🌲:
'''
count = 0

path='/home/mati/code/Mati-Gonc/Earth-Project/screenshots/test1.png'

place_to_find = st.text_input('Quel ville ?')

if place_to_find and count == 0:
    launch_search(place_to_find, path)
    image = Image.open(path)
    image_cropped = image.crop((120, 120, 900, 900))
    image_cropped = image_cropped.resize((2448, 2448), resample=0)
    st.image(image_cropped)
    rgb_im = image_cropped.convert('RGB')
    imgArray = np.array(rgb_im)

    res = None
    count=1

if st.button('predict brof'):
    # Send to API, endpoint must accept POST
    endpoint = 'http://127.0.0.1:8000/predict'
    # Ensure json content type
    headers = {}
    headers['Content-Type'] = 'application/json'
    # Use helpers method to prepare image for request
    request_dict = image_to_dict(imgArray)
    #print(len(request_dict))
    # Post image data, and get prediction
    res = rq.post(endpoint, json.dumps(request_dict), headers=headers).json()
    if res:
        st.image(image_from_dict(res['image']))
