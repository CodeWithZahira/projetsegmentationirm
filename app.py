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
image_url = "https://static.vecteezy.com/system/resources/previews/012/847/554/large_2x/minimal-background-purple-color-with-two-overlapping-waves-suitable-for-design-needs-display-website-ui-and-others-free-photo.jpg"
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
        
