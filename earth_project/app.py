import streamlit as st

import streamlit as st
import numpy as np
from PIL import Image
import requests as rq
import json
from image_to_API import image_to_dict, image_from_dict
import io

'''
# Earth's project 🛰 🔎 🗺
'''

col1,col2 = st.columns(2)
col1.markdown("### Enter a city")
place_to_find = col1.text_input('in France')

col2.markdown("### Or, upload your file")
uploaded_file = col2.file_uploader("Everywhere in the world ", type=["png", "jpg", "jpeg"])
res = None


if place_to_find:
    url = f'https://image.maps.ls.hereapi.com/mia/1.6/mapview?apiKey=EGJw9wPvG7b8taiDqr88xgPMA68nwJxtwybySFulHaE&sb=km&w=2448&h=2448&z=15&co=france&ci={place_to_find}&t=1&ppi=3'
    response = rq.get(url)
    if response.status_code != 200:
        print('something is broken bro, and it is because of you')
    in_memory_file = io.BytesIO(response.content)
    image = Image.open(in_memory_file)
    image = image.resize((2448, 2448), resample=0)
    st.image(image)
    rgb_im = image.convert('RGB')
    imgArray = np.array(rgb_im)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image)
    rgb_im = image.convert('RGB')
    imgArray = np.array(rgb_im)

if st.button("Let's the magic happen"):
    # Send to API, endpoint must accept POST
    endpoint = 'https://luneapi-6vuwhckoyq-ew.a.run.app/predict'
    # Ensure json content type
    headers = {}
    headers['Content-Type'] = 'application/json'
    # Use helpers method to prepare image for request
    request_dict = image_to_dict(imgArray)
    print(len(request_dict))
    # Post image data, and get prediction
    res = rq.post(endpoint, json.dumps(request_dict), headers=headers).json()
    if res:
        array_return = image_from_dict(res['image'])
        tot3 = np.repeat(array_return, 3, axis=3)
        tot3 = np.where(tot3 == [0, 0, 0], [170, 170, 170], tot3)
        tot3 = np.where(tot3 == [1, 1, 1], [179, 0, 12], tot3)
        tot3 = np.where(tot3 == [2, 2, 2], [0, 127, 8], tot3)
        tot3 = np.where(tot3 == [3, 3, 3], [62, 120, 208], tot3)

        st.image(tot3)
