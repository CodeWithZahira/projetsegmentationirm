import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import io
import streamlit.components.v1 as com

# =============================
# 📦 UTILITY FUNCTIONS
# =============================

def preprocess_image(image_file, target_size=(128, 128)):
    image = Image.open(image_file).convert("L")
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
        st.image(image_pil, caption="Original MRI Scan", use_container_width=True)
    with col2:
        st.image(mask, caption="Predicted Segmentation Mask", use_container_width=True)

# =============================
# 🔧 PAGE CONFIG
# =============================
st.set_page_config(page_title="NeuroSeg Interactive", layout="wide")

# =============================
# 🎨 STYLING & ASSETS (Consolidated)
# =============================
image_url = "https://4kwallpapers.com/images/wallpapers/3d-background-glass-light-abstract-background-blue-3840x2160-8728.jpg"
st.markdown(f"""
<style>
/* CSS Variable for Animation */
@property --a {{
  syntax: "<angle>";
  initial-value: 0deg;
  inherits: false;
}}

/* General App and Font Styling */
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
    background: linear-gradient(45deg, rgba(255,255,255,0.7), rgba(255,255,255,0.7));
    z-index: -1;
}}

/* Set all text to black */
h1, h2, h3, h4, h5, h6, p, span, div, .stMarkdown, .stFileUploader label, .stButton button, .stLinkButton button, .st-emotion-cache-1c7y2kd, .st-emotion-cache-1v0mbdj, .custom-footer-box, .custom-footer-box * {{
    color: black !important;
}}

/* Animated Button Styling */
.animated-button-container {{
    position: relative;
    display: inline-block;
    padding: 3px;
    border-radius: 50px;
    overflow: hidden;
    width: 100%;
    text-align: center;
}}
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
  to {{ --a: 1turn; }}
}}
.animated-button-container .stButton>button, .st-emotion-cache-19n6j20 a {{
    width: 100%;
    background: linear-gradient(45deg, #005c97, #363795);
    color: black !important;
    border-radius: 50px;
    padding: 15px 30px;
    font-size: 18px;
    font-weight: bold;
    border: none;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    transition: all 0.3s ease;
    text-decoration: none;
}}
.animated-button-container .stButton>button:hover, .st-emotion-cache-19n6j20 a:hover {{
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.4);
}}

/* --- Custom Footer Styling (FIXED) --- */
.custom-footer-box {{
    background: rgba(255, 255, 255, 0.6);
    padding: 30px;
    border-radius: 15px;
    border: 1px solid #00c6ff;
    margin-top: 4rem;
    overflow: auto;
}}
.custom-footer-box h4 {{
    margin-bottom: 10px;
}}
.custom-footer-box a {{
    color: #007acc !important;
    text-decoration: none;
}}
.custom-footer-box img {{
    border-radius: 8px;
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
            "<h1 style='text-align: center; font-family: sans-serif; font-weight: 800; font-size: 3.5rem;'>NeuroSeg</h1>",
            unsafe_allow_html=True
        )
        st.markdown(
            "<p style='text-align: center; font-size:1.5rem;'>Witness the future of medical imaging. Upload your model and MRI scan to experience the power of AI-driven segmentation.</p>",
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

    st.markdown(f'<div class="animated-button-container">{st.link_button("⬇️ Download the Model (.tflite)", model_download_url, use_container_width=True)}</div>', unsafe_allow_html=True)

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
        st.image(image_file, caption="Uploaded MRI Scan", use_container_width=True)

if model_loaded and image_file:
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="animated-button-container">', unsafe_allow_html=True)
    if st.button("🔍 Perform Segmentation", use_container_width=True):
        with st.spinner('Analyzing the image...'):
            img_array, img_pil = preprocess_image(image_file)
            pred_mask = tflite_predict(interpreter, img_array)
            display_prediction(img_pil, pred_mask)
    st.markdown('</div>', unsafe_allow_html=True)

# =============================
# 🎓 ABOUT & CREDITS FOOTER
# =============================
st.markdown('<div class="custom-footer-box">', unsafe_allow_html=True)

logo_url = "https://tse2.mm.bing.net/th/id/OIP.WC5xs7MJrmfk_YEHDn6BOAAAAA?pid=Api&P=0&h=180"

f_col1, f_col2 = st.columns([1, 2])
with f_col1:
    st.markdown(f'<div style="text-align: center;"><img src="{logo_url}" width="150"></div>', unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'>Cadi Ayyad University<br>Faculties of Medicine and Pharmacie</p>", unsafe_allow_html=True)
with f_col2:
    st.markdown(
        """
        <div>
            <h4>Developed By</h4>
            <p>ELLAOUAH ZAHIRA | <a href="mailto:zahiraellaouah@gmail.com">zahiraellaouah@gmail.com</a></p>
            <h4>Under the Supervision of</h4>
            <p>Pr. Nezha Oumghar      Pr. Mohamed Amine Chadi</p>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown('</div>', unsafe_allow_html=True)
