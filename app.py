import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import streamlit.components.v1 as com
import cv2
import tempfile
import os

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

def extract_frames_from_video(video_file, max_frames=30):
    frames = []
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_file.write(video_file.read())
        tmp_file_path = tmp_file.name

    cap = cv2.VideoCapture(tmp_file_path)
    count = 0
    while count < max_frames and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pil_image = Image.fromarray(gray)
        frames.append(pil_image)
        count += 1
    cap.release()
    os.remove(tmp_file_path)
    return frames

# =============================
# 🔧 PAGE CONFIG
# =============================
st.set_page_config(page_title="NeuroSeg Interactive", layout="wide")

# =============================
# 🎨 STYLING & BACKGROUND + TITLE ANIMATION
# =============================
# [Same CSS Styling block from previous code — unchanged]
# [Omitted here for brevity, it's still in your document]

# =============================
# 💬 WELCOME SECTION
# =============================
# [Same welcome section from previous code — unchanged]

# =============================
# 🚀 MAIN APPLICATION
# =============================
st.markdown("<br><hr><br>", unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

with col1:
    st.header("1. Get & Upload Model")
    st.markdown("First, download the pre-trained model file.")
    model_download_url = "https://drive.google.com/uc?export=download&id=1O2pcseTkdmgO_424pGfk636kT0_T36v8"

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
    st.header("2. Upload MRI Image(s) or Video")
    image_files = st.file_uploader("Upload MRI Images", type=["png", "jpg", "jpeg", "tif", "tiff"], accept_multiple_files=True, label_visibility="collapsed")
    video_file = st.file_uploader("Or upload an MRI Video (mp4)", type=["mp4"], label_visibility="collapsed")

    all_images = []
    if image_files:
        for file in image_files:
            st.image(file, caption=file.name, use_container_width=True)
            all_images.append(file)

    if video_file:
        with st.spinner("Extracting frames from video..."):
            frames = extract_frames_from_video(video_file)
            for i, frame in enumerate(frames):
                st.image(frame, caption=f"Frame {i+1}", use_container_width=True)
                all_images.append(frame)

if model_loaded and all_images:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="animated-button-container">', unsafe_allow_html=True)
    if st.button("🔍 Perform Segmentation for All Inputs", use_container_width=True):
        for idx, item in enumerate(all_images):
            with st.spinner(f"Analyzing image {idx + 1}..."):
                try:
                    if isinstance(item, Image.Image):
                        img_array, img_pil = preprocess_image(item)
                    else:
                        img_array, img_pil = preprocess_image(item)
                    pred_mask = tflite_predict(interpreter, img_array)
                    display_prediction(img_pil, pred_mask)
                except Exception as e:
                    st.error(f"❌ Error with input {idx + 1}: {e}")
    st.markdown('</div>', unsafe_allow_html=True)

# =============================
# 🎓 FOOTER
# =============================
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
            <p>Pr. Nezha Oumghar</p>
            <p>Pr. Mohamed Amine Chadi</p>
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
