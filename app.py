import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from streamlit_echarts import st_echarts
import numpy as np
import json
import requests
import matplotlib.pyplot as plt
import random
from quickdraw import QuickDrawData

st.set_page_config(layout="wide")
st.sidebar.header('A.I. Pictionary')
st.sidebar.image("lewagon.png")
st.sidebar.write("Amsterdam - Batch 627")
st.sidebar.write("#")

st.sidebar.write(
    "[Maroussia Loots](https://www.linkedin.com/in/maroussia-loots)")

st.sidebar.write(
    "[Andreas Mussger](https://www.linkedin.com/in/andreasmussger/)")

st.sidebar.write(
    "[Hidde Uittenbosch](https://www.linkedin.com/in/hidde-uittenbosch/)")

st.sidebar.write("#")
st.sidebar.write("[GitHub repository](https://github.com/a-mger/ai_pictionary)")



labels_250 = np.load(f'CNN_250_labels.npy')
labels_250 = labels_250.tolist()
col1, col2, col3 = st.beta_columns((2,3,3))


col1.header('Choose a model')
with col1:
    choose_model = st.selectbox(
    '50 Labels / 150 Labels / 250 Labels',
    ('CNN_50', 'CNN_150', 'CNN_250'))
    if st.button('Show 5 random labels!'):
        labels = np.load(f'{choose_model}_labels.npy')
        fivelabels = random.sample(range(0, len(labels)), 5)
        for i in fivelabels:
            st.write(labels[i])

col2.header('Draw Something')
col3.header('A.I. Prediction')

with col2:
    realtimeupdate = st.checkbox('Update in real time', False)
    canvas_result = st_canvas(
        stroke_width=8.2,
        stroke_color="rgba(0, 0, 0,1)",
        update_streamlit=realtimeupdate,
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

with col3:


    if canvas_result.image_data is not None:
        img = Image.fromarray(canvas_result.image_data.astype(np.uint8))
        img_28_28 = img.resize((28, 28), Image.ANTIALIAS)
        img_28_28 = np.asarray(img_28_28)
        img_gray = 255 - img_28_28[:, :, 3]
        img_gray = np.invert(img_gray)
        nocanvas = (img_gray.sum() == 0)

        if nocanvas != True:
            img_gray[img_gray < 50] = 0
            img_gray[(img_gray > 51) & (img_gray < 100)] = 75
            img_gray[(img_gray > 101) & (img_gray < 149)] = 149
            img_gray[img_gray > 150] = 255


            img_list = img_gray.tolist()
            img_json = json.dumps(img_list)
            # enter here the address of your flask api
            url = 'https://aipictionaryimage-djqbxeaiha-ew.a.run.app/predict'
            #url = 'http://127.0.0.1:8000/predict'
            params = dict(img_frontend=img_json, model_type=choose_model)
            response = requests.get(url, params=params)
            prediction = response.json()
            df = pd.read_json(prediction)
            fig, ax = plt.subplots()

            options = {

                "legend": {
                    "top": "5%",
                    "left": "center"
                },
                "series": [{
                    "name":
                    "Probability",
                    "type":
                    "pie",
                    "radius": ["40%", "70%"],
                    "avoidLabelOverlap":
                    False,
                    "itemStyle": {
                        "borderRadius": 10,
                        "borderColor": "#fff",
                        "borderWidth": 2,
                    },
                    "label": {
                        "show": False,
                        "position": "center"
                    },
                    "emphasis": {
                        "label": {
                            "show": True,
                            "fontSize": "40",
                            "fontWeight": "bold"
                        }
                    },
                    "labelLine": {
                        "show": False
                    },
                    "data": [
                        {
                            "value":
                            round(df.values[0][0] * 100),
                            "name":
                            f"{df.columns[0]} {round(df.values[0][0] * 100)}%"
                        },
                        {
                            "value":
                            round(df.values[0][1] * 100),
                            "name":
                            f"{df.columns[1]} {round(df.values[0][1] * 100)}%"
                        },
                        {
                            "value":
                            round(df.values[0][2] * 100),
                            "name":
                            f"{df.columns[2]} {round(df.values[0][2] * 100)}%"
                        },
                        {
                            "value":
                            round(df.values[0][3] * 100),
                            "name":
                            f"{df.columns[3]} {round(df.values[0][3] * 100)}%"
                        },
                        {
                            "value":
                            round(df.values[0][4] * 100),
                            "name":
                            f"{df.columns[4]} {round(df.values[0][4] * 100)}%"
                        },
                    ],
                }],
            }
            st_echarts(options=options, height="500px")
        else:
            st.image("cartoon.jpeg")
    else:
        st.image("cartoon.jpeg")
st.write("#")
st.write("#")
st.write("#")
col11, col21, col31 = st.beta_columns((2, 3, 3))

with col11:

    pic_request = st.text_input('Paste a Label and show how other people drew it')
    if st.button('Provide Picture'):
        if pic_request in labels_250:
            qd = QuickDrawData()
            st.image(qd.get_drawing(pic_request).image)
        elif pic_request == '':
            st.write('#')
        else:
            st.write('Not a valid label')
