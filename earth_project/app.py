import streamlit as st
import numpy as np
from PIL import Image
import requests as rq
import json
from image_to_API import image_to_dict, image_from_dict
import io

'''
# Badass forest calculator 🔎🌲:
'''
place_to_find = st.text_input('================================== Tu veux voir la forêt de quelle ville bro ? =================================', value= 'terter')
uploaded_file = st.file_uploader("============================== Vazy, tu peux direct donner une image si tu veux ============================== ", type=["png", "jpg", "jpeg"])
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

if st.button('predict brof'):
    # Send to API, endpoint must accept POST
    endpoint = 'https://earthboite-6vuwhckoyq-od.a.run.app/predict'
    # Ensure json content type
    headers = {}
    headers['Content-Type'] = 'application/json'
    # Use helpers method to prepare image for request
    request_dict = image_to_dict(imgArray)
    print(len(request_dict))
    # Post image data, and get prediction
    print(endpoint)
    print(request_dict)
    print(headers)
    res = rq.post(endpoint, json.dumps(request_dict), headers=headers).json()
    if res:
        st.image(image_from_dict(res['image']))
