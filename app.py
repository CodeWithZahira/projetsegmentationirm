import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
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
# 🎨 STYLING & BACKGROUND
# =============================
image_url = "https://images.pexels.com/photos/7130560/pexels-photo-7130560.jpeg"
st.markdown(f"""
<style>
/* CSS Variable for other animations */
@property --a {{
  syntax: "<angle>";
  initial-value: 0deg;
  inherits: false;
}}

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
    /* Using a darker overlay to make white text pop */
    background: linear-gradient(45deg, rgba(15, 32, 39, 0.8), rgba(32, 58, 67, 0.9));
    z-index: -1;
}}

/* Set default text color to white for better readability on dark overlay */
h1, h2, h3, h4, h5, h6, p, span, div, .stMarkdown, .stFileUploader label {{
    color: white !important;
}}

/* --- ✨ NEW Title Animation --- */
.title-animation {{
    font-size: 4rem; /* Making title a bit bigger */
    font-weight: 800;
    font-family: sans-serif;
    text-align: center;
    background: linear-gradient(to right, #7997e8, #f6d3ff, #8aa9f7, #f6d3ff);
    background-size: 200% auto;
    color: #000;
    background-clip: text;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: animateGradient 5s linear infinite;
}}
@keyframes animateGradient {{
    to {{
        background-position: 200% center;
    }}
}}
/* --- End of Title Animation --- */


/* Animated Button */
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
.animated-button-container .stButton>button, .animated-button-container .st-emotion-cache-19n6j20 a {{
    width: 100%;
    background: linear-gradient(45deg, #005c97, #363795);
    color: white !important; /* Button text is white */
    border-radius: 50px;
    padding: 15px 30px;
    font-size: 18px;
    font-weight: bold;
    border: none;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    transition: all 0.3s ease;
    text-decoration: none;
}}
.animated-button-container .stButton>button:hover, .animated-button-container .st-emotion-cache-19n6j20 a:hover {{
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.4);
}}

/* Footer Styling */
.booking-style-footer {{
    background-color: #0d1117; /* Dark footer to match theme */
    padding: 50px 30px 20px 30px;
    font-family: sans-serif;
    border-top: 1px solid #30363d;
    color: #c9d1d9; /* Light grey text */
}}
.booking-style-footer h4 {{
    font-size: 18px;
    margin-bottom: 10px;
    font-weight: bold;
    color: white;
}}
.booking-style-footer p, .booking-style-footer a {{
    font-size: 15px;
    color: #c9d1d9;
    text-decoration: none;
    margin: 4px 0;
}
.booking-style-footer a:hover {{
    text-decoration: underline;
}
.footer-columns {{
    display: flex;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 50px;
}
.footer-column {{
    flex: 1;
    min-width: 200px;
}
.footer-bottom {{
    margin-top: 30px;
    text-align: center;
    font-size: 13px;
    color: #8b949e;
    border-top: 1px solid #30363d;
    padding-top: 15px;
}
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
        # Applying the new .title-animation class to the h1 tag
        st.markdown(
            "<h1 class='title-animation'>NeuroSeg</h1>",
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
    model_download_url = "https://drive.google.com/uc?export=download&id=1O2pcseTkdmgO_424pGfk636kT0_T36v8"

    # Wrapping the link_button in the animation container
    st.markdown('<div class="animated-button-container">', unsafe_allow_html=True)
    st.link_button("⬇️ Download the Model (.tflite)", model_download_url, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

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
    # Wrapping the regular button in the animation container
    st.markdown('<div class="animated-button-container">', unsafe_allow_html=True)
    if st.button("🔍 Perform Segmentation", use_container_width=True):
        with st.spinner('Analyzing the image...'):
            img_array, img_pil = preprocess_image(image_file)
            pred_mask = tflite_predict(interpreter, img_array)
            display_prediction(img_pil, pred_mask)
    st.markdown('</div>', unsafe_allow_html=True)

# =============================
# 🎓 BOOKING.COM STYLE FOOTER
# =============================
st.markdown("""
<div class="booking-style-footer">
    <div class="footer-columns">
        <div class="footer-column">
            <h4>Developed By</h4>
            <p>Zahira ELLAOUAH</p>
            <p><a href="mailto:zahiraellaouah@gmail.com">zahiraellaouah@gmail.com</a></p>
        </div>
        <div class="footer-column">
            <h4>Supervised By</h4>
            <p>Prof. Nezha Oumghar</p>
            <p>Prof. Mohamed Amine Chadi</p>
        </div>
        <div class="footer-column">
            <h4>University</h4>
            <p>Cadi Ayyad University</p>
            <p>Faculty of Medicine and Pharmacy</p>
            <p>Marrakesh</p>
        </div>
        <div class="footer-column">
            <h4>Project</h4>
            <p>Automatic Segmentation of Brain MRIs by Convolutional Neural Network U-Net</p>
            <p>Master's in biomedical instrumentation and analysis</p>
        </div>
    </div>
    <div class="footer-bottom">
        <p>© 2025 Zahira Ellaouah – All rights reserved</p>
    </div>
</div>
""", unsafe_allow_html=True)
