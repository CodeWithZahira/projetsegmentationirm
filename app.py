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
image_url = "https://images.pexels.com/photos/691668/pexels-photo-691668.jpeg"

# --- Combined CSS Block ---
st.markdown(f"""
<style>
/* --- Google Font Import --- */
@import url('https://fonts.googleapis.com/css2?family=Roboto&display=swap');

/* --- CSS Variable for Button Animation --- */
@property --a {{
  syntax: "<angle>";
  initial-value: 0deg;
  inherits: false;
}}

/* --- Base Body & Font Styling --- */
body {{
  font-family: 'Roboto' !important; 
  color: white;
}}

/* --- Background Image and Overlay --- */
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
    z-index: 0; /* Behind content but above background */
}}

/* --- Magic Canvas for Particle Effects --- */
#magic {{
  position: fixed;
  width: 100%;
  height: 100vh;
  display: block;
  top: 0;
  left: 0;
  z-index: -1; /* Behind everything */
}}

/* --- NEW WELCOME SCREEN STYLES --- */
.playground {{
  height: 100vh; /* Full viewport height */
  display: flex;
  flex-direction: column;
  justify-content: flex-end; /* Align content to the bottom */
  align-items: center;
  text-align: center;
  padding-bottom: 50px;
}}
.playground h1 {{
    font-size: 3.5rem;
    font-weight: 800;
}}
.playground .minText {{
  font-size: 1.2rem;
  color: #ccc;
  max-width: 600px;
}}
.scroll-prompt {{
    font-size: 2rem;
    margin-top: 2rem;
    animation: bounce 2s infinite;
}}
@keyframes bounce {{
    0%, 20%, 50%, 80%, 100% {{ transform: translateY(0); }}
    40% {{ transform: translateY(-20px); }}
    60% {{ transform: translateY(-10px); }}
}}


/* --- ANIMATED BUTTON STYLES --- */
.animated-button-container {{
    position: relative;
    display: inline-block;
    padding: 3px;
    border-radius: 50px;
    overflow: hidden;
    width: 100%;
    text-align: center;
    margin-top: 1rem;
}}
.animated-button-container::before {{
    content: "";
    position: absolute;
    z-index: 0;
    inset: -0.5em;
    border: solid 0.25em;
    border-image: conic-gradient(from var(--a), #7997e8, #f6d3ff, #7997e8) 1;
    filter: blur(0.25em);
    animation: rotateGlow 4s linear infinite;
}}
@keyframes rotateGlow {{ to {{ --a: 1turn; }} }}

.animated-button-container .stButton>button, .animated-button-container .stLinkButton>a {{
    width: 100%;
    background: linear-gradient(45deg, #005c97, #363795);
    color: white !important;
    border-radius: 50px;
    padding: 15px 30px;
    font-size: 18px;
    font-weight: bold;
    border: none;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    transition: all 0.3s ease;
    position: relative;
    z-index: 1;
}}
.animated-button-container .stButton>button:hover, .animated-button-container .stLinkButton>a:hover {{
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.4);
}}


/* --- FOOTER SECTION STYLE --- */
.footer-container {{
    background: rgba(15, 32, 39, 0.8);
    padding: 2rem;
    border-radius: 10px;
    margin-top: 4rem;
    border-top: 1px solid #00c6ff;
}}
.footer-container .footer {{ color: #ccc; text-align: center; }}
.footer-container .footer a {{ color: #00c6ff; text-decoration: none; }}

</style>
""", unsafe_allow_html=True)


# =============================
# 💬 WELCOME SCREEN
# =============================
# --- Canvas for future JS animations ---
st.markdown('<canvas id="magic"></canvas>', unsafe_allow_html=True)

# --- New Welcome "Playground" Layout ---
st.markdown("""
<div class="playground">
    <div style="width: 200px; margin-bottom: 2rem;">
        <img src="https://lottie.host/embed/a0bb04f2-9027-4848-907f-e4891de977af/lnTdVRZOiZ.json" style="width:100%; display:none;">
    </div>
    <h1>NeuroSeg</h1>
    <p class="minText">
        An immersive journey into the future of medical imaging. 
        Upload your model and MRI scan to experience the power of AI-driven segmentation.
    </p>
    <p class="scroll-prompt">↓</p>
</div>
""", unsafe_allow_html=True)
# The Lottie animation is tricky to embed directly here with the iframe,
# so we'll use a placeholder and focus on the layout. If needed, the com.iframe can be placed here.
# For now, the clean text layout is prioritized.

# =============================
# 🚀 MAIN APPLICATION
# =============================
# (The rest of your application code remains the same)
st.markdown("<br><hr><br>", unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

with col1:
    st.header("1. Get & Upload Model")
    st.markdown("First, download the pre-trained model file.")
    model_download_url = "https://drive.google.com/uc?export=download&id=YOUR_FILE_ID_HERE"
    
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
