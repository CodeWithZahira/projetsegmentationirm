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
        
