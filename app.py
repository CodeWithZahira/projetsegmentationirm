import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import io
import streamlit.components.v1 as com

# =============================
# 🔧 PAGE CONFIG
# =============================
st.set_page_config(page_title="NeuroSeg Interactive", layout="wide")

# =============================
# 🎨 STYLING & ASSETS
# =============================
# --- Background Image ---
image_url = "data:image/jpeg;base64,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"

# --- Main CSS for Background, Fonts, and the NEW Button Animation ---
st.markdown(f"""
<style>
/* Registering the CSS variable for animation */
@property --a {{
  syntax: "<angle>";
  initial-value: 0deg;
  inherits: false;
}}

/* Main Background Image and Overlay */
.stApp {{
    background-image: url("{image_url}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}}
.stApp::before {{
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: linear-gradient(45deg, rgba(15, 32, 39, 0.9), rgba(32, 58, 67, 0.9), rgba(44, 83, 100, 0.9));
    z-index: -1;
}}

/* General Text Color */
h1, h2, h3, h4, h5, h6, p, .stMarkdown, .stFileUploader label {{
    color: #FFFFFF !important;
}}

/* --- NEW ANIMATED BUTTON STYLE --- */
/* We create a container to hold the animation */
.animated-button-container {{
    position: relative;
    display: inline-block;
    padding: 3px; /* Space for the border to show */
    border-radius: 50px; /* Match the button's border-radius */
    overflow: hidden;
    width: 100%;
    text-align: center;
}}

/* The glowing, rotating border effect */
.animated-button-container::before {{
    content: "";
    position: absolute;
    z-index: -1;
    inset: -0.5em;
    border: solid 0.25em;
    border-image: conic-gradient(from var(--a), #7997e8, #f6d3ff, #7997e8) 1;
    filter: blur(0.25em);
    animation: rotateGlow 4s linear infinite;
}}

@keyframes rotateGlow {{
  to {{
    --a: 1turn;
  }}
}}

/* Styling the actual Streamlit button inside the container */
.animated-button-container .stButton>button, .animated-button-container .stLinkButton>a {{
    width: 100%;
    background: linear-gradient(45deg, #005c97, #363795);
    color: white;
    border-radius: 50px;
    padding: 15px 30px;
    font-size: 18px;
    font-weight: bold;
    border: none;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    transition: all 0.3s ease;
}}
.animated-button-container .stButton>button:hover, .animated-button-container .stLinkButton>a:hover {{
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.4);
}}


/* --- NEW FOOTER SECTION STYLE --- */
.footer-container {{
    background: rgba(15, 32, 39, 0.8); /* Semi-transparent dark background */
    padding: 2rem;
    border-radius: 10px;
    margin-top: 4rem;
    border-top: 1px solid #00c6ff; /* A nice top border to separate it */
}}
.footer-container .footer {{
    color: #ccc;
    text-align: center;
}}
.footer-container .footer a {{
    color: #00c6ff;
    text-decoration: none;
}}
</style>
""", unsafe_allow_html=True)


# =============================
# 💬 WELCOME SECTION
# =============================
with st.container():
    col1, col2 = st.columns([1, 2], gap="large")
    with col1:
        com.iframe(
            "https://lottie.host/embed/a0bb04f2-9027-4848-907f-e4891de977af/lnTdVRZOiZ.lottie",
            height=400
        )
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown(
            "<h1 style='text-align: center; color: #fff; font-family: sans-serif; font-weight: 800; font-size: 3.5rem;'>NeuroSeg</h1>",
            unsafe_allow_html=True
        )
        st.markdown(
            "<p style='text-align: center; color:#ccc; font-size:1.5rem;'>Witness the future of medical imaging. Upload your model and MRI scan to experience the power of AI-driven segmentation.</p>",
            unsafe_allow_html=True
        )


# =============================
# 🚀 MAIN APPLICATION
# =============================
st.markdown("<br><hr><br>", unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

with col1:
    st.header("1. Get & Upload Model")
    st.markdown("First, download the pre-trained model file.")
    model_download_url = "https://drive.google.com/uc?export=download&id=YOUR_FILE_ID_HERE"
    
    # --- Applying the animation to the Link Button ---
    st.markdown(f'<div class="animated-button-container"><a href="{model_download_url}" target="_blank" class="stLinkButton" style="display: block; text-decoration: none; color: white; padding: 15px 30px; border-radius: 50px; background: linear-gradient(45deg, #005c97, #363795);">⬇️ Download the Model (.tflite)</a></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("Then, upload the downloaded file here:")
    model_file = st.file_uploader("Upload model", type=["tflite"], label_visibility="collapsed")
    
    interpreter = None
    model_loaded = False
    if model_file:
        try:
            tflite_model = model_file.read()
            interpreter = tf.lite.Interpreter(model_content=tflite_model)
            interpreter.allocate_tensors()
            st.success("✅ Model loaded successfully.")
            model_loaded = True
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")

with col2:
    st.header("2. Upload Image")
    st.markdown("Now, upload an MRI scan to perform segmentation.")
    image_file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg", "tif", "tiff"], label_visibility="collapsed")
    if image_file:
        st.image(image_file, caption="Uploaded MRI Scan", use_column_width=True)

if model_loaded and image_file:
    st.markdown("<br>", unsafe_allow_html=True)
    
    # --- Applying the animation to the regular Button ---
    st.markdown('<div class="animated-button-container">', unsafe_allow_html=True)
    if st.button("🔍 Perform Segmentation"):
        with st.spinner('Analyzing the image...'):
            img_array, img_pil = preprocess_image(Image.open(image_file))
            pred_mask = tflite_predict(interpreter, img_array)
            display_prediction(img_pil, pred_mask)
    st.markdown('</div>', unsafe_allow_html=True)

# =============================
# 🎓 ABOUT & CREDITS FOOTER
# =============================
# --- Applying the new footer container class ---
st.markdown('<div class="footer-container">', unsafe_allow_html=True)

logo_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0c/MIT_logo.svg/1200px-MIT_logo.svg.png"

f_col1, f_col2 = st.columns([1, 2])
with f_col1:
    st.markdown(f'<div style="text-align: center; padding-top: 20px;"><img src="{logo_url}" width="100"></div>', unsafe_allow_html=True)
    st.markdown("<p class='footer'>[Your University Name]</p>", unsafe_allow_html=True)
with f_col2:
    st.markdown(
        """
        <div class="footer">
            <h4>Developed By</h4>
            <p>👤 [Your Name Here] | <a href="mailto:[your.email@university.edu]">📧 [your.email@university.edu]</a></p>
            <h4>Under the Supervision of</h4>
            <p>👨‍🏫 [Professor 1 Name]     👨‍🏫 [Professor 2 Name]</p>
        </div>
        """,
        unsafe_allow_html=True
    )
st.markdown('</div>', unsafe_allow_html=True)


# =============================
# 📦 UTILITY FUNCTIONS
# =============================
# (Your utility functions remain the same)
def preprocess_image(uploaded_file, target_size=(128, 128)):
    image = uploaded_file.convert("L")
    image = image.resize(target_size)
    img_array = np.array(image) / 255.0
    img_array = img_array.astype(np.float32)
    img_array = np.expand_dims(img_array, axis=(0, -1))
    return img_array, image

def tflite_predict(interpreter, input_data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    prediction = output_data[0, :, :, 0]
    prediction = (prediction > 0.5).astype(np.uint8) * 255
    return prediction

def display_prediction(image_pil, mask):
    st.markdown("---")
    st.subheader("Segmentation Result")
    col1, col2 = st.columns(2)
    with col1:
        st.image(image_pil, caption="Original MRI Scan", use_column_width=True)
    with col2:
        st.image(mask, caption="Predicted Segmentation Mask", use_column_width=True)
        
