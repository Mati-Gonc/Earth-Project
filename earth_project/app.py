from sys import displayhook
import streamlit as st

import streamlit as st
import numpy as np
from PIL import Image
import requests as rq
import json
from image_to_API import image_to_dict, image_from_dict

'''
# Coloss forest calculator 🔎🌲:
'''

uploaded_file = st.file_uploader("Gimme  image", type=["png", "jpg", "jpeg"])
res = None

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image)
    rgb_im = image.convert('RGB')
    imgArray = np.array(rgb_im)
if st.button('predict brof'):
    # Send to API, endpoint must accept POST
    endpoint = 'http://127.0.0.1:8000/predict'
    # Ensure json content type
    headers = {}
    headers['Content-Type'] = 'application/json'
    # Use helpers method to prepare image for request
    request_dict = image_to_dict(imgArray)
    print(len(request_dict))
    # Post image data, and get prediction
    res = rq.post(endpoint, json.dumps(request_dict), headers=headers).json()
    if res:
        st.image(image_from_dict(res['image']))
