import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
from PIL import Image
import sys
import base64

np.set_printoptions(threshold=sys.maxsize)

# Specify canvas parameters in application
# stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
# stroke_color = st.sidebar.color_picker("Stroke color hex: ")
# bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
# bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])
# drawing_mode = st.sidebar.selectbox(
#     "Drawing tool:", ("freedraw", "line", "rect", "circle", "transform"))
# realtime_update = st.sidebar.checkbox("Update in realtime", True)

# Create a canvas component
canvas_result = st_canvas(
    #fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=3, #to play around with or adapt through testing model
    stroke_color="rgba(0, 0, 0,1)",
    #background_color="" if bg_image else bg_color,
    #background_image=Image.open(bg_image) if bg_image else None,
    update_streamlit=True,
    height=400,
    width=400,
    drawing_mode="freedraw",
    key="canvas",
)
# Do something interesting with the image data and paths

if canvas_result.json_data is not None:
    st.dataframe(pd.json_normalize(canvas_result.json_data["objects"]))
    #df = pd.json_normalize(canvas_result.json_data["objects"])["path"]

if canvas_result.image_data is not None:
    #st.image(canvas_result.image_data)
    img = Image.fromarray(canvas_result.image_data.astype(np.uint8))
    img_28_28 = img.resize((28, 28), Image.ANTIALIAS)
    img_28_28 = np.asarray(img_28_28)
    img_gray = 255 - img_28_28[:, :, 3]
    img_gray = np.invert(img_gray)
    img_gray = np.reshape(img_gray, (784, 1))
    print("##################################")
    print(img_gray)
    #st.image(img_gray)
    st.write(img_gray)
    