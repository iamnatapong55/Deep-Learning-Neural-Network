import streamlit as st
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline
import logging

logging.getLogger("torch.nn.functional").setLevel(logging.ERROR)

@st.cache(allow_output_mutation=True)
def load_image_generation_model():
    model = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token="TOKEN").to("cuda" if torch.cuda.is_available() else "cpu")
    return model

model = load_image_generation_model()

# Custom theme settings
st.markdown(
    """
    <style>
    .css-18e3th9 {
        background-color: #FFF8F0;
        color: #4A4A4A;
    }
    .css-1d391kg {
        background-color: #FFDAB9;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #6D4C41;
    }
    .stButton>button {
        color: #FFF8F0;
        border-radius: 20px;
        border: 1px solid #FFDAB9;
        background-color: #D2691E;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("AI-Powered Room Decoration App")
st.write("Upload an image of an empty room and customize the design based on your preferences.")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
room_type = st.selectbox('Room type:', ('Living Room', 'Bedroom', 'Kitchen', 'Bathroom', 'Office'))
color_preference = st.selectbox('Color preference:', ('Vibrant and Bright', 'Soft and Pastel', 'Dark and Moody', 'Earthy and Natural', 'Bold and Dramatic'))
design_style = st.selectbox('Design style:', ('Modern', 'Classic', 'Bohemian', 'Rustic', 'Industrial', 'Minimalist', 'Scandinavian', 'Art Deco'))
furniture_style = st.selectbox('Furniture style:', ('Sleek and Simple', 'Ornate and Traditional', 'Functional and Minimal', 'Rustic and Wooden', 'Innovative and Modern'))

if st.button('Design Room') and uploaded_file:
    with st.spinner('Crafting your personalized room design... Please wait.'):
        detailed_prompt = f"Enhance the original {room_type.lower()} photo to appear more realistic with high-quality textures, natural lighting, {color_preference.lower()} color palettes, {design_style.lower()} design aesthetics, and {furniture_style.lower()} style furniture."
        image = Image.open(uploaded_file).convert("RGB")
        output = model(
            prompt=detailed_prompt, 
            init_image=image, 
            num_inference_steps=30,  
            guidance_scale=5.5  
        ).images[0]
        st.image(output, caption='Your Customized Room Design')
else:
    st.warning("Please upload an image to start the design process.")



















