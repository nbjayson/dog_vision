import streamlit as st
from code import make_prediction
from PIL import Image
import base64

st.markdown(
    f"""
    <style>
    .reportview-container {{
        background: url(data:image/.jpg;base64,{base64.b64encode(open('app/images/bg.jpg', "rb").read()).decode()})
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Dog breed classification app")
st.subheader("Upload an image of a dog and let's predict the breed")

uploaded_file = st.file_uploader("Choose a dog photo...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded image.', use_column_width=True)
    st.markdown("")
    st.markdown("### Oh, isn't he a cute boy")
    st.markdown("### Let's see...")
    label = make_prediction(image)
    st.markdown(f'### Uhm, well, the dog is probably a **{label[0]}**')
    st.markdown(f"### P.S: I'm sorry if I got it wrong :(")