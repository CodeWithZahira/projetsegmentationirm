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
# 🎨 STYLING & BACKGROUND + TITLE ANIMATION
# =============================
image_url = "https://4kwallpapers.com/images/wallpapers/3d-background-glass-light-abstract-background-blue-3840x2160-8728.jpg"
st.markdown(f"""
<style>
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
    background: linear-gradient(45deg, rgba(255,255,255,0.7), rgba(255,255,255,0.7));
    z-index: -1;
}}

h1, h2, h3, h4, h5, h6, p, span, div, .stMarkdown, .stFileUploader label, .stButton button, .stLinkButton button {{
    color: black !important;
}}

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
.animated-button-container .stButton>button {{
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
}}
.animated-button-container .stButton>button:hover {{
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.4);
}}

@keyframes glowBounce {{
  0%, 100% {{
    color: #005c97;
    text-shadow:
      0 0 5px #7997e8,
      0 0 10px #7997e8,
      0 0 20px #7997e8,
      0 0 40px #f6d3ff,
      0 0 80px #f6d3ff;
    transform: translateY(0) scale(1);
  }}
  50% {{
    color: #f6d3ff;
    text-shadow:
      0 0 10px #f6d3ff,
      0 0 20px #f6d3ff,
      0 0 30px #f6d3ff,
      0 0 60px #7997e8,
      0 0 90px #7997e8;
    transform: translateY(-20px) scale(1.15);
  }}
}}

.animated-title {{
  font-family: 'Roboto', sans-serif;
  font-weight: 900;
  font-size: 5rem;
  text-align: center;
  animation: glowBounce 2.5s ease-in-out infinite;
  user-select: none;
  margin-bottom: 0.5rem;
  cursor: default;
}}
</style>
""", unsafe_allow_html=True)

# =============================
# 💬 WELCOME SECTION
# =============================
st.markdown('<h1 class="animated-title">NeuroSeg</h1>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size:1.5rem;'>Upload your model and MRI images or videos to explore AI-powered segmentation in action.</p>", unsafe_allow_html=True)

# =============================
# MODEL & IMAGE UPLOAD
# =============================
col1, col2 = st.columns(2)

with col1:
    st.header("1. Upload Model")
    model_file = st.file_uploader("Upload .tflite model", type=["tflite"])
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
    st.header("2. Upload MRI Images")
    image_files = st.file_uploader("Upload one or more images", type=["jpg", "jpeg", "png", "tif"], accept_multiple_files=True)
    if image_files:
        for file in image_files:
            st.image(file, caption=file.name, use_container_width=True)

# =============================
# 📽️ VIDEO UPLOAD & FRAME EXTRACTION
# =============================
st.markdown("### 📽️ Or Upload an MRI Video")
video_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"], key="video")
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
# SEGMENTATION SECTION
# =============================
if model_loaded:
    if image_files:
        st.markdown("<h3>🔬 Segment Uploaded Images</h3>", unsafe_allow_html=True)
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
        st.markdown("<h3>🎞️ Segment Extracted Frames</h3>", unsafe_allow_html=True)
        if st.button("Segment Frames"):
            for idx, frame_img in enumerate(video_frames):
                with st.spinner(f"Segmenting frame {idx + 1}..."):
                    try:
                        img_array, processed_img = preprocess_image(frame_img)
                        pred_mask = tflite_predict(interpreter, img_array)
                        display_prediction(processed_img, pred_mask)
                    except Exception as e:
                        st.error(f"❌ Error with frame {idx + 1}: {e}")

# =============================
# FOOTER
# =============================
st.markdown("""
---
<p style="text-align:center;font-size:14px;">© 2025 Zahira Ellaouah · Supervised by Pr. Nezha Oumghar & Pr. Mohamed Amine Chadi · Cadi Ayyad University – FMPM</p>
""", unsafe_allow_html=True)
