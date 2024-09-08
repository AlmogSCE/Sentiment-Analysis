import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from utils import resize_image

st.title("Sentiment Analysis Dashboard")
st.subheader("Dataset")

st.write(pd.read_csv('./assets/partial.csv', index_col=0))

base_path = './assets/output'
image_paths = [f'{base_path}.png', f'{base_path}1.png', f'{base_path}2.png', f'{base_path}3.png', f'{base_path}4.png', f'{base_path}5.png', ]
images = [resize_image(Image.open(image_path), target_height=300) for image_path in image_paths]

if 'image_index' not in st.session_state:
    st.session_state.image_index = 0

st.subheader("Graphs")

# Display the current image
st.image(images[st.session_state.image_index], use_column_width=True)

col1, col2, col3 = st.columns([1, 12, 1])
with col1:
    if st.button("<"):
        st.session_state.image_index = (st.session_state.image_index - 1) % len(images)
with col3:
    if st.button("‎ >"):
        st.session_state.image_index = (st.session_state.image_index + 1) % len(images)