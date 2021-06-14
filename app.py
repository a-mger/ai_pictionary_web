import pandas as pd
from PIL import Image
import streamlit as st
from streamlit.caching import cache
from streamlit.elements import button
from streamlit_drawable_canvas import st_canvas
from streamlit_echarts import st_echarts
import numpy as np
import json
import requests
import seaborn as sns
import matplotlib.pyplot as plt
import random
st. set_page_config(layout="wide")

col1, col2, col3 = st.beta_columns((2,3,3))

col1.header('Wanna play a game?')


@st.cache(suppress_st_warning=True)
def fivelabels(num):
    #button = st.button('Press me!')
    labels = np.load('labels.npy')
    fivelabels = random.sample(range(0, len(labels)), num)
    for i in fivelabels:
        st.write(labels[i])

with col1:
    if st.button('Press me!'):
        labels = np.load('labels.npy')
        fivelabels = random.sample(range(0, len(labels)), 5)
        for i in fivelabels:
            st.write(labels[i])
                
col2.header('Start drawing')
col3.header('Most likely drawing')

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
with col2:
    realtimeupdate = st.checkbox('Update in real time', False)
    canvas_result = st_canvas(
        #fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=8.2,
        stroke_color="rgba(0, 0, 0,1)",
        #background_color="" if bg_image else bg_color,
        #background_image=Image.open(bg_image) if bg_image else None,
        update_streamlit=realtimeupdate,
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )
    
# Do something interesting with the image data and paths
#if canvas_result.json_data is not None:
#st.dataframe(pd.json_normalize(canvas_result.json_data["objects"]))
#df = pd.json_normalize(canvas_result.json_data["objects"])["path"]
with col3:
    
    if canvas_result.image_data is not None:
        img = Image.fromarray(canvas_result.image_data.astype(np.uint8))
        img_28_28 = img.resize((28, 28), Image.ANTIALIAS)
        img_28_28 = np.asarray(img_28_28)
        img_gray = 255 - img_28_28[:, :, 3]
        img_gray = np.invert(img_gray)
        nocanvas = (img_gray.sum() == 0)
    
        if nocanvas != True:
            #st.image(canvas_result.image_data)
            print(img_gray)
            print('----'*100)
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
            #ax = sns.barplot(y=df.values[0], x=df.columns)
            #st.pyplot(fig)
            #st_echarts(type=pie, )
            
            options = {
                "tooltip": {"trigger": "item"},
                "legend": {"top": "5%", "left": "center"},
                "series": [
                    {
                        "name": "Probability",
                        "type": "pie",
                        "radius": ["40%", "70%"],
                        "avoidLabelOverlap": False,
                        "itemStyle": {
                            "borderRadius": 10,
                            "borderColor": "#fff",
                            "borderWidth": 2,
                        },
                        "label": {"show": False, "position": "center"},
                        "emphasis": {
                            "label": {"show": True, "fontSize": "40", "fontWeight": "bold"}
                        },
                        "labelLine": {"show": False},
                        "data": [
                            {"value": round(df.values[0][0] * 100), "name": df.columns[0]},
                            {"value": round(df.values[0][1] * 100), "name": df.columns[1]},
                            {"value": round(df.values[0][2] * 100), "name": df.columns[2]},
                            {"value": round(df.values[0][3] * 100), "name": df.columns[3]},
                            {"value": round(df.values[0][4] * 100), "name": df.columns[4]},
                        ],
                    }
                ],
            }
            st_echarts(options=options, height="500px")