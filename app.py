import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import json
import requests
import seaborn as sns
import matplotlib.pyplot as plt

# Specify canvas parameters in application
# stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
# stroke_color = st.sidebar.color_picker("Stroke color hex: ")
# bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
# bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])
# drawing_mode = st.sidebar.selectbox(
#     "Drawing tool:", ("freedraw", "line", "rect", "circle", "transform"))
# realtime_update = st.sidebar.checkbox("Update in realtime", True)
#
# Create a canvas component
canvas_result = st_canvas(
    #fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=8.2,
    stroke_color="rgba(0, 0, 0,1)",
    #background_color="" if bg_image else bg_color,
    #background_image=Image.open(bg_image) if bg_image else None,
    update_streamlit=True,
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)
# Do something interesting with the image data and paths
#if canvas_result.json_data is not None:
#st.dataframe(pd.json_normalize(canvas_result.json_data["objects"]))
#df = pd.json_normalize(canvas_result.json_data["objects"])["path"]
if canvas_result.image_data is not None:
    #st.image(canvas_result.image_data)
    img = Image.fromarray(canvas_result.image_data.astype(np.uint8))
    img_28_28 = img.resize((28, 28), Image.ANTIALIAS)
    img_28_28 = np.asarray(img_28_28)
    img_gray = 255 - img_28_28[:, :, 3]
    img_gray = np.invert(img_gray)
    
    #optimizing images 
    img_gray[img_gray < 50] = 0
    img_gray[(img_gray > 51) & (img_gray < 100)] = 75
    img_gray[(img_gray > 101) & (img_gray < 149)] = 149
    img_gray[img_gray > 150] = 255
    
    img_list = img_gray.tolist()
    img_json = json.dumps(img_list)

    # enter here the address of your flask api
    url = 'https://aipictionaryimage-djqbxeaiha-ew.a.run.app/predict'
    #url = 'http://127.0.0.1:8000/predict'
    params = dict(img_frontend=img_json)
    response = requests.get(url, params=params)
    prediction = response.json()
    df = pd.read_json(prediction)
    fig, ax = plt.subplots()
    ax = sns.barplot(y=df.values[0], x=df.columns)
    st.pyplot(fig, height=300, width = 300)
