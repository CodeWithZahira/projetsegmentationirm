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
st.set_page_config(page_title="NeuroSeg Advanced", layout="wide")

# =============================
# 🎨 STYLING & ASSETS
# =============================
# Replace this URL with one for your desired background image
image_url = "https://images.pexels.com/photos/691668/pexels-photo-691668.jpeg" 

page_bg_img = f"""
<style>
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
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(45deg, rgba(15, 32, 39, 0.85), rgba(32, 58, 67, 0.85), rgba(44, 83, 100, 0.85));
    z-index: -1;
}}
/* Footer styling */
.footer {{
    color: #ccc;
    text-align: center;
}}
.footer a {{
    color: #00c6ff;
    text-decoration: none;
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

st.markdown("""
<style>
/* General Font and Color Styles */
h1, h2, h3, h4, h5, h6, p, .stMarkdown, .stFileUploader, .stLinkButton {
    color: #FFFFFF !important;
}

/* Custom Button Style */
.stButton>button {
    background: linear-gradient(45deg, #00c6ff, #0072ff);
    color: white;
    border-radius: 50px;
    padding: 15px 30px;
    font-size: 18px;
    font-weight: bold;
    border: none;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    transition: all 0.3s ease;
}

.stButton>button:hover {
    transform: translateY(-3px);
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
    background: linear-gradient(45deg, #0072ff, #00c6ff);
}

/* Custom File Uploader Style */
.stFileUploader label {
    font-size: 1.2em;
    font-weight: bold;
    color: #FFFFFF;
}
</style>
""", unsafe_allow_html=True)


# =============================
# 💬 Bienvenue + Animation
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
# 📦 UTILITIES
# =============================
def preprocess_image(uploaded_file, target_size=(128, 128)):
    image = Image.open(uploaded_file).convert("L")
    image = image.resize(target_size)
    img_array = np.array(image) / 255.0
    img_array = img_array.astype(np.float32)
    img_array = np.expand_dims(img_array, axis=(0, -1))
    return img_array, image

def tflite_predict(interpreter, input_data):
    # (Your function remains the same)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    prediction = output_data[0, :, :, 0]
    prediction = (prediction > 0.5).astype(np.uint8) * 255
    return prediction

def display_prediction(image_pil, mask):
    # (Your function remains the same)
    st.markdown("---")
    st.subheader("Segmentation Result")
    col1, col2 = st.columns(2)
    with col1:
        st.image(image_pil, caption="Original MRI Scan", use_column_width=True)
    with col2:
        st.image(mask, caption="Predicted Segmentation Mask", use_column_width=True)

# =============================
# 🚀 MAIN APPLICATION
# =============================
st.markdown("<br><hr><br>", unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

with col1:
    st.header("1. Get & Upload Model")
    st.markdown("First, download the pre-trained model file.")
    
    # --- PASTE YOUR GOOGLE DRIVE DIRECT DOWNLOAD LINK HERE ---
    model_download_url = "https://drive.google.com/uc?export=download&id=YOUR_FILE_ID_HERE" 
    
    st.link_button("⬇️ Download the Model (.tflite)", model_download_url)
    
    st.markdown("---")
    st.markdown("Then, upload the downloaded file here:")
    model_file = st.file_uploader("📁 Import your .tflite model", type=["tflite"], label_visibility="collapsed")
    
    model_loaded = False
    interpreter = None
    if model_file is not None:
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
    image_file = st.file_uploader("🖼️ Import an MRI image (PNG/JPG/TIF)", type=["png", "jpg", "jpeg", "tif", "tiff"], label_visibility="collapsed")
    if image_file:
        st.image(image_file, caption="Uploaded MRI Scan", use_column_width=True)

if model_loaded and image_file:
    st.markdown("<br>", unsafe_allow_html=True)
    col_center, _, _ = st.columns([1,1,1])
    with col_center:
        if st.button("🔍 Perform Segmentation"):
            with st.spinner('Analyzing the image...'):
                img_array, img_pil = preprocess_image(image_file)
                pred_mask = tflite_predict(interpreter, img_array)
                display_prediction(img_pil, pred_mask)

# =============================
# 🎓 ABOUT & CREDITS FOOTER
# =============================
st.markdown("<br><br><br><hr>", unsafe_allow_html=True)

# --- Replace with your university's logo URL and details ---
logo_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0c/MIT_logo.svg/1200px-MIT_logo.svg.png" # Example: MIT Logo

with st.container():
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
