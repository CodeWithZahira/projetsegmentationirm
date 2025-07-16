import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import streamlit.components.v1 as com

# =============================
# 📆 PAGE CONFIG
# =============================
st.set_page_config(page_title="NeuroSeg Interactive", layout="wide")

# =============================
# 🌟 TOP BAR (Logo, Search, About)
# =============================
st.markdown(f"""
<style>
.topbar {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 20px;
    background-color: rgba(255, 255, 255, 0.8);
    border-radius: 12px;
    margin-bottom: 10px;
}}

.topbar-logo {{
    height: 45px;
}}

.topbar-search {{
    flex-grow: 1;
    margin: 0 20px;
}}

.topbar-search input {{
    width: 100%;
    padding: 8px 12px;
    font-size: 14px;
    border: 2px solid #005c97;
    border-radius: 8px;
}}

.topbar-about {{
    padding: 8px 18px;
    background: linear-gradient(45deg, #005c97, #363795);
    color: white;
    border-radius: 8px;
    font-weight: bold;
    text-decoration: none;
}}
</style>

<div class="topbar">
    <img class="topbar-logo" src="https://upload.wikimedia.org/wikipedia/fr/thumb/7/76/Université_Cadi_Ayyad.svg/1200px-Université_Cadi_Ayyad.svg.png">
    <div class="topbar-search">
        <input type="text" placeholder="🔍 Research..." />
    </div>
    <a href="#about-section" class="topbar-about">About</a>
</div>
""", unsafe_allow_html=True)

# =============================
# 📚 HEADER TITLE SECTION
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

st.markdown("<br><hr><br>", unsafe_allow_html=True)
col1, col2 = st.columns(2, gap="large")

with col1:
    st.header("1. Get & Upload Model")
    model_url = "https://drive.google.com/uc?export=download&id=1O2pcseTkdmgO_424pGfk636kT0_T36v8"
    st.link_button("⬇️ Download Model (.tflite)", model_url, use_container_width=True)
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
    image_file = st.file_uploader("Upload MRI scan", type=["png", "jpg", "jpeg", "tif", "tiff"], label_visibility="collapsed")
    if image_file:
        st.image(image_file, caption="Uploaded MRI Scan", use_container_width=True)

if model_loaded and image_file:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔍 Perform Segmentation", use_container_width=True):
        with st.spinner('Analyzing the image...'):
            img_array, img_pil = preprocess_image(image_file)
            pred_mask = tflite_predict(interpreter, img_array)
            display_prediction(img_pil, pred_mask)

# =============================
# 📄 ABOUT SECTION (ANCHOR)
# =============================
st.markdown('<br><br><hr><br>', unsafe_allow_html=True)
st.markdown('<h2 id="about-section">👩‍💼 About the Project</h2>', unsafe_allow_html=True)

st.markdown("""
<style>
.booking-style-footer {
    background-color: #f9f9f9;
    padding: 50px 30px 20px 30px;
    font-family: sans-serif;
    border-top: 1px solid #ddd;
    color: black;
}
.booking-style-footer h4 {
    font-size: 18px;
    margin-bottom: 10px;
    font-weight: bold;
}
.booking-style-footer p, .booking-style-footer a {
    font-size: 15px;
    color: black;
    text-decoration: none;
    margin: 4px 0;
}
.footer-columns {
    display: flex;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 50px;
}
.footer-column {
    flex: 1;
    min-width: 200px;
}
.footer-bottom {
    margin-top: 30px;
    text-align: center;
    font-size: 13px;
    color: #666;
    border-top: 1px solid #ddd;
    padding-top: 15px;
}
</style>

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
            <p>Automatic Segmentation of Brain MRIs by CNN U-Net</p>
            <p>Master's in Biomedical Instrumentation and Analysis</p>
        </div>
    </div>
    <div class="footer-bottom">
        <p>© 2025 Zahira Ellaouah – All rights reserved</p>
    </div>
</div>
""", unsafe_allow_html=True)
