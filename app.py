import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
import streamlit.components.v1 as com
import cv2
import tempfile
import os
from io import BytesIO
from fpdf import FPDF
import zipfile

# =============================
# 📦 UTILITY FUNCTIONS
# =============================

def preprocess_image(image_file, target_size=(128, 128)):
    if isinstance(image_file, Image.Image):
        image = image_file.convert("L")
    else:
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

def overlay_mask_on_image(image_pil, mask, color=(255, 0, 0), alpha=0.4):
    image_np = np.array(image_pil.convert("RGB"))
    mask_rgb = np.zeros_like(image_np)
    mask_rgb[mask == 255] = color
    blended = cv2.addWeighted(image_np, 1, mask_rgb, alpha, 0)
    return Image.fromarray(blended)

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

def generate_combined_downloads(all_results):
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zipf:
        for idx, (orig, mask, overlay) in enumerate(all_results):
            # Save overlay as PNG
            img_bytes = BytesIO()
            overlay.save(img_bytes, format="PNG")
            zipf.writestr(f"segmented_{idx+1}.png", img_bytes.getvalue())

            # Save overlay as PDF
            pdf = FPDF()
            pdf.add_page()
            pdf_w, pdf_h = pdf.w - 2*pdf.l_margin, pdf.h - 2*pdf.t_margin
            buffered_pdf_img = BytesIO()
            overlay.save(buffered_pdf_img, format="JPEG")
            buffered_pdf_img.seek(0)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_img_file:
                tmp_img_file.write(buffered_pdf_img.read())
                tmp_img_file_path = tmp_img_file.name
            pdf.image(tmp_img_file_path, x=pdf.l_margin, y=pdf.t_margin, w=pdf_w)
            pdf_bytes = pdf.output(dest="S").encode("latin1")
            zipf.writestr(f"segmented_{idx+1}.pdf", pdf_bytes)
            os.remove(tmp_img_file_path)

    zip_buffer.seek(0)
    st.download_button(
        label="📦 Download All Segmentations (PNG + PDF)",
        data=zip_buffer,
        file_name="all_segmented_results.zip",
        mime="application/zip"
    )

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
.stApp {{
    background-image: url("{image_url}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}}
.animated-title {{
  font-family: 'Roboto', sans-serif;
  font-weight: 900;
  font-size: 5rem;
  text-align: center;
  animation: glowBounce 2.5s ease-in-out infinite;
  margin-bottom: 0.5rem;
  cursor: default;
}}
@keyframes glowBounce {{
  0%, 100% {{
    color: #005c97;
    text-shadow: 0 0 10px #7997e8, 0 0 20px #7997e8, 0 0 30px #7997e8;
    transform: translateY(0) scale(1);
  }}
  50% {{
    color: #f6d3ff;
    text-shadow: 0 0 20px #f6d3ff, 0 0 40px #f6d3ff, 0 0 60px #7997e8;
    transform: translateY(-10px) scale(1.05);
  }}
}}
</style>
""", unsafe_allow_html=True)

# =============================
# 💬 WELCOME SECTION
# =============================

st.markdown('<h1 class="animated-title">NeuroSeg</h1>', unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; font-size:1.5rem;'>Upload your model and MRI scan(s) or video to experience the power of AI-driven segmentation.</p>",
    unsafe_allow_html=True
)

# =============================
# 🚀 MAIN APPLICATION
# =============================

col1, col2 = st.columns(2)

with col1:
    st.header("1. Upload Model")
    model_file = st.file_uploader("Upload your TFLite model", type=["tflite"])
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
    image_files = st.file_uploader("Upload MRI Images", type=["png", "jpg", "jpeg", "tif", "tiff"], accept_multiple_files=True)
    video_file = st.file_uploader("Or upload a video (mp4 or avi)", type=["mp4", "avi"])

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

# =============================
# 🔍 Perform Segmentation
# =============================

if model_loaded and all_images:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="animated-button-container">', unsafe_allow_html=True)
    if st.button("🔍 Perform Segmentation", use_container_width=True):
        all_results = []
        for idx, item in enumerate(all_images):
            with st.spinner(f"Processing input {idx + 1}..."):
                try:
                    if isinstance(item, Image.Image):
                        img_array, img_pil = preprocess_image(item)
                    else:
                        img_array, img_pil = preprocess_image(item)
                    pred_mask = tflite_predict(interpreter, img_array)

                    overlay_img = overlay_mask_on_image(img_pil, pred_mask)

                    st.image(overlay_img, caption=f"Overlay MRI + Mask {idx+1}", use_container_width=True)

                    all_results.append((img_pil, pred_mask, overlay_img))

                except Exception as e:
                    st.error(f"❌ Error with input {idx + 1}: {e}")

        generate_combined_downloads(all_results)

    st.markdown('</div>', unsafe_allow_html=True)

# =============================
# 🎓 FOOTER
# =============================

st.markdown("""
<hr style='margin-top: 50px;'>
<div style='text-align: center; font-size: 14px; color: gray;'>
    Developed by Zahira Ellaouah · Supervised by Pr. Nezha Oumghar & Pr. Mohamed Amine Chadi · FMPM - Cadi Ayyad University
</div>
""", unsafe_allow_html=True)
