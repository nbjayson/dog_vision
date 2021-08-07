import streamlit as st
from code import predict_custom
from PIL import Image

st.title('Dog Vision')

uploaded_file = st.file_uploader("Choose a dog photo...", type=["jpg", "jpeg", "png"])
def load_image(image_file):
	img = Image.open(image_file)
	return img

if uploaded_file is not None:
	uploaded_file = load_image(uploaded_file)
	image, label, confidence = predict_custom(uploaded_file)

	st.image(image=image, caption=f'Breed: {label}, Confidence: {confidence}')