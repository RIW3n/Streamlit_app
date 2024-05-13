import streamlit as st
from ultralytics import YOLO
from PIL import Image
model = YOLO("yolov8n.pt")

with st.sidebar:
    st.text("Settings")
    thresh = st.slider("Threshold",min_value=0.40,max_value=0.99,value=0.5)
with st.expander("About this app"):
    st.text("""YOLO powered Streamlit app
            For object Detection""")

img_file = st.file_uploader("Bruv upload your image",type=['png','jpg'],help="this should only be images.") 

if img_file:
    col1,col2 = st.columns(2)
    col1.image(img_file,use_column_width=True,caption="Original")
    # st.image(img_file,caption="this is your image.")
    Image.open(img_file).save(img_file.name)
    results = model(img_file.name,stream=False,conf= thresh)
    results[0].save(filename ='results.jpg')
    # st.image("results.jpg",caption="this is your image.")
    col2.image("results.jpg",use_column_width=True,caption="Prediction")
