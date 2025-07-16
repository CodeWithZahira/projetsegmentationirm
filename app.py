import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import io
import streamlit.components.v1 as com
import base64

# =============================
# 🔧 PAGE CONFIG
# =============================
st.set_page_config(page_title="NeuroSeg 3D", layout="wide")

# =============================
# 🎨 DYNAMIC ASSETS & STYLING
# =============================

# --- STEP 1: ADD YOUR BACKGROUND VIDEO ---
# Download a video and place it in your project folder, e.g., "background.mp4"
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

try:
    video_base64 = get_base64_of_bin_file("background.mp4")
except FileNotFoundError:
    st.warning("`background.mp4` not found. Please download a background video to your project folder.")
    video_base64 = ""

page_bg_video = f"""
<style>
.stApp {{
    background: #000; /* Fallback color */
}}
#bg-video {{
    position: fixed;
    top: 50%;
    left: 50%;
    min-width: 100%;
    min-height: 100%;
    width: auto;
    height: auto;
    z-index: -2;
    transform: translateX(-50%) translateY(-50%);
}}
/* Overlay */
.stApp::before {{
    content: "";
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(45deg, rgba(15, 32, 39, 0.85), rgba(32, 58, 67, 0.85), rgba(44, 83, 100, 0.85));
    z-index: -1;
}}
</style>
<video autoplay loop muted id="bg-video">
    <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
</video>
"""
st.markdown(page_bg_video, unsafe_allow_html=True)


# Custom CSS for buttons and fonts
st.markdown("""
<style>
/* Your existing CSS for buttons, fonts, etc. */
h1, h2, h3, h4, h5, h6, p, .stMarkdown, .stFileUploader, .stLinkButton { color: #FFFFFF !important; }
.stButton>button {
    background: linear-gradient(45deg, #00c6ff, #0072ff); color: white; border-radius: 50px;
    padding: 15px 30px; font-size: 18px; font-weight: bold; border: none;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2); transition: all 0.3s ease;
}
.stButton>button:hover {
    transform: translateY(-3px); box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
    background: linear-gradient(45deg, #0072ff, #00c6ff);
}
.stFileUploader label { font-size: 1.2em; font-weight: bold; color: #FFFFFF; }
.footer { color: #ccc; text-align: center; }
.footer a { color: #00c6ff; text-decoration: none; }
</style>
""", unsafe_allow_html=True)


# =============================
# 📜 PAGE CONTENT - SCROLLING SECTIONS
# =============================

# --- SECTION 1: WELCOME BANNER ---
with st.container():
    st.markdown("<div style='height:100vh; display:flex; flex-direction:column; justify-content:center; align-items:center;'>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; font-size: 4.5rem; font-weight: 800;'>NeuroSeg 3D</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.5rem; max-width: 800px;'>An immersive journey into AI-driven medical imaging. Scroll down to begin.</p>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 2rem; margin-top: 2rem;'>↓</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# --- SECTION 2: EMBEDDED 3D MODEL ---
with st.container():
    st.markdown("<div style='height:100vh; display:flex; flex-direction:column; justify-content:center; align-items:center;'>", unsafe_allow_html=True)
    st.markdown("## Explore the Subject", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'>Interact with the 3D model below. Drag to rotate and scroll to zoom.</p>", unsafe_allow_html=True)
    
    # --- STEP 2: PASTE YOUR SKETCHFAB IFRAME SRC HERE ---
    sketchfab_url = "https://sketchfab.com/models/b042a1b322394560a34657744317426a/embed" # Example: Brain model
    
    com.iframe(sketchfab_url, height=500, scrolling=False)
    st.markdown("<p style='text-align: center; font-size: 2rem; margin-top: 2rem;'>↓</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# --- SECTION 3: THE MAIN APPLICATION ---
with st.container():
    st.markdown("---")
    st.header("🧠 Perform Segmentation")
    st.markdown("<p style='text-align:center;'>Now, it's your turn. Provide the model and an MRI scan to see the magic happen.</p>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.subheader("1. Get & Upload Model")
        st.markdown("First, download the pre-trained model file.")
        model_download_url = "https://drive.google.com/uc?export=download&id=YOUR_FILE_ID_HERE"
        st.link_button("⬇️ Download the Model (.tflite)", model_download_url)
        st.markdown("Then, upload the downloaded file here:")
        model_file = st.file_uploader("📁 Import model", type=["tflite"], label_visibility="collapsed")
        
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
        st.subheader("2. Upload Image")
        st.markdown("Now, upload an MRI scan.")
        image_file = st.file_uploader("🖼️ Import image", type=["png", "jpg", "jpeg", "tif", "tiff"], label_visibility="collapsed")
        if image_file:
            st.image(image_file, caption="Uploaded MRI Scan", use_column_width=True)

    if model_loaded and image_file:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔍 Perform Segmentation"):
            with st.spinner('Analyzing the image...'):
                img_array, img_pil = preprocess_image(Image.open(image_file), target_size=(128, 128))
                pred_mask = tflite_predict(interpreter, img_array)
                display_prediction(img_pil, pred_mask)


# --- SECTION 4: CREDITS FOOTER ---
st.markdown("<div style='height:50vh;'></div>", unsafe_allow_html=True) # Spacer
with st.container():
    st.markdown("<hr>", unsafe_allow_html=True)
    # Your footer code here (logo, names, etc.)
    logo_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0c/MIT_logo.svg/1200px-MIT_logo.svg.png" 
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(f'<div style="text-align: center; padding-top: 20px;"><img src="{logo_url}" width="100"></div>', unsafe_allow_html=True)
        st.markdown("<p class='footer'>[Your University Name]</p>", unsafe_allow_html=True)
    with col2:
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

# Your utility functions (preprocess_image, tflite_predict, display_prediction) remain the same
# Make sure they are included at the end of the script or in the utilities section.
# ... (rest of your utility functions)
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
