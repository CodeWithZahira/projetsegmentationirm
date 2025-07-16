import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import streamlit.components.v1 as com
import cv2
import tempfile

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
# 🎨 TITLE AND DESCRIPTION
# =============================
st.markdown('<h1 style="text-align: center; font-size: 3.5rem;">NeuroSeg</h1>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size:1.2rem;'>Upload your model and MRI scan(s) or video to experience AI-driven segmentation.</p>", unsafe_allow_html=True)

# =============================
# 🚀 MAIN APPLICATION
# =============================
col1, col2 = st.columns(2, gap="large")

with col1:
    st.header("1. Upload Model")
    model_file = st.file_uploader("Upload your .tflite model", type=["tflite"])
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
    st.header("2. Upload MRI Image(s)")
    image_files = st.file_uploader("Upload MRI Images", type=["png", "jpg", "jpeg", "tif", "tiff"], accept_multiple_files=True)
    if image_files:
        for file in image_files:
            st.image(file, caption=file.name, use_container_width=True)

# =============================
# 📽️ VIDEO SECTION
# =============================
st.markdown("### 📽️ Or Upload MRI Video")
video_file = st.file_uploader("Upload MRI video", type=["mp4", "avi", "mov"], key="video")
video_frames = []

if video_file is not None:
    st.video(video_file)
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(video_file.read())
        tmp_path = tmp_file.name

    cap = cv2.VideoCapture(tmp_path)
    success, frame = cap.read()
    frame_count = 0

    while success and frame_count < 20:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pil_img = Image.fromarray(gray)
        video_frames.append(pil_img)
        success, frame = cap.read()
        frame_count += 1
    cap.release()

    st.success(f"✅ Extracted {len(video_frames)} frames from video.")
    for idx, frame in enumerate(video_frames):
        st.image(frame, caption=f"Frame {idx+1}", use_container_width=True)

# =============================
# 🔍 SEGMENTATION SECTION
# =============================
if model_loaded:
    if image_files:
        st.markdown("<hr><h3>🔬 Segment Uploaded Images</h3>", unsafe_allow_html=True)
        if st.button("Segment Images"):
            for idx, image_file in enumerate(image_files):
                with st.spinner(f"Segmenting image {idx + 1}..."):
                    try:
                        img_array, img_pil = preprocess_image(image_file)
                        pred_mask = tflite_predict(interpreter, img_array)
                        display_prediction(img_pil, pred_mask)
                    except Exception as e:
                        st.error(f"❌ Error with image {image_file.name}: {e}")

    if video_frames:
        st.markdown("<hr><h3>🎞️ Segment Video Frames</h3>", unsafe_allow_html=True)
        if st.button("Segment Video Frames"):
            for idx, frame_img in enumerate(video_frames):
                with st.spinner(f"Segmenting frame {idx + 1}..."):
                    try:
                        img_array, processed_img = preprocess_image(frame_img)
                        pred_mask = tflite_predict(interpreter, img_array)
                        display_prediction(processed_img, pred_mask)
                    except Exception as e:
                        st.error(f"❌ Error with frame {idx + 1}: {e}")

# =============================
# 👣 FOOTER
# =============================
st.markdown("""
---
**Developed by Zahira Ellaouah | Supervised by Pr. Nezha Oumghar & Pr. Mohamed Amine Chadi**  
Faculty of Medicine and Pharmacy, Cadi Ayyad University – Master's in Biomedical Instrumentation
""")
