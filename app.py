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

def superimpose_mask_on_image(original_pil, mask_np, mask_color=(255, 0, 0), alpha=0.4):
    """Superimpose a colored mask on a grayscale image."""
    mask_pil = Image.fromarray(mask_np).resize(original_pil.size)
    original_rgb = original_pil.convert("RGB")
    color_mask = Image.new("RGB", original_pil.size, mask_color)
    mask_rgba = mask_pil.convert("L").point(lambda p: int(p * alpha))
    composite = Image.composite(color_mask, original_rgb, mask_rgba)
    return composite

def display_prediction(image_pil, mask):
    st.markdown("---")
    st.subheader("Segmentation Result")
    col1, col2 = st.columns(2)
    with col1:
        st.image(image_pil, caption="Original MRI Scan", use_container_width=True)
    with col2:
        superimposed = superimpose_mask_on_image(image_pil, mask, mask_color=(255, 0, 0), alpha=0.4)
        st.image(superimposed, caption="Superimposed Segmentation Mask", use_container_width=True)

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

def combine_images(original: Image.Image, mask: np.ndarray) -> Image.Image:
    mask_pil = Image.fromarray(mask).convert("L").resize(original.size)
    combined_width = original.width + mask_pil.width
    combined_height = max(original.height, mask_pil.height)
    combined_img = Image.new("RGB", (combined_width, combined_height))
    original_rgb = original.convert("RGB")
    combined_img.paste(original_rgb, (0, 0))
    combined_img.paste(mask_pil.convert("RGB"), (original.width, 0))
    draw = ImageDraw.Draw(combined_img)
    font = ImageFont.load_default()
    draw.text((10, 10), "MRI Scan", fill="red", font=font)
    draw.text((original.width + 10, 10), "Segmentation Mask", fill="red", font=font)
    return combined_img

def get_combined_download_links(original, mask, idx):
    combined_img = combine_images(original, mask)
    buffered_png = BytesIO()
    combined_img.save(buffered_png, format="PNG")
    st.download_button(label="📥 Download MRI + Mask as PNG", data=buffered_png.getvalue(), file_name=f"combined_{idx+1}.png", mime="image/png")
    
    pdf = FPDF()
    pdf.add_page()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_img_file:
        combined_img.save(tmp_img_file, format="JPEG")
        pdf.image(tmp_img_file.name, x=pdf.l_margin, y=pdf.t_margin, w=pdf.w - 2*pdf.l_margin)
    pdf_output = pdf.output(dest="S").encode("latin1")
    st.download_button(label="📥 Download MRI + Mask as PDF", data=pdf_output, file_name=f"combined_{idx+1}.pdf", mime="application/pdf")

# =============================
# 🔧 PAGE CONFIG
# =============================
st.set_page_config(page_title="NeuroSeg Interactive", layout="wide")

# =============================
# 🎨 STYLING & BACKGROUND + TITLE ANIMATION
# =============================
# (All the styling CSS remains the same as before)
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
h1, h2, h3, h4, h5, h6, p, span, div, .stMarkdown, .stFileUploader label, .stButton button, .stLinkButton button, .st-emotion-cache-1c7y2kd, .st-emotion-cache-1v0mbdj {{
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
    text-shadow: 0 0 5px #7997e8, 0 0 10px #7997e8, 0 0 20px #7997e8, 0 0 40px #f6d3ff, 0 0 80px #f6d3ff;
    transform: translateY(0) scale(1);
  }}
  50% {{
    color: #f6d3ff;
    text-shadow: 0 0 10px #f6d3ff, 0 0 20px #f6d3ff, 0 0 30px #f6d3ff, 0 0 60px #7997e8, 0 0 90px #7997e8;
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
with st.container():
    col1, col2 = st.columns([1, 2], gap="large")
    with col1:
        com.iframe("https://lottie.host/embed/a0bb04f2-9027-4848-907f-e4891de977af/lnTdVRZOiZ.lottie", height=400)
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown('<h1 class="animated-title">NeuroSeg</h1>', unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; font-size:1.5rem;'>Witness the future of medical imaging. Upload your model and MRI scan(s) or video to experience the power of AI-driven segmentation.</p>", unsafe_allow_html=True)

# =============================
# 🚀 MAIN APPLICATION
# =============================
st.markdown("<br><hr><br>", unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

with col1:
    st.header("1. Get & Upload Model")
    st.markdown("First, download the pre-trained model file.")
    model_download_url = "https://drive.google.com/uc?export=download&id=1HPkPPcdUD-bcOUnwhlNf_UpQG5OBy8qr"
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
    
    #<-- START OF MODIFIED SECTION -->
    # Initialize session state for file uploaders if they don't exist
    if 'image_uploader' not in st.session_state:
        st.session_state.image_uploader = []
    if 'video_uploader' not in st.session_state:
        st.session_state.video_uploader = None

    # Use the 'key' argument for the file uploaders
    image_files = st.file_uploader("Upload MRI Images", type=["png", "jpg", "jpeg", "tif", "tiff"], accept_multiple_files=True, label_visibility="collapsed", key="image_uploader")
    video_file = st.file_uploader("Or upload an MRI Video (mp4 or avi)", type=["mp4", "avi"], label_visibility="collapsed", key="video_uploader")

    # When the button is clicked, clear the state and force a rerun
    if st.button("🧹 Clear All Images & Videos", use_container_width=True):
        st.session_state.image_uploader = []
        st.session_state.video_uploader = None
        st.rerun()
    #<-- END OF MODIFIED SECTION -->

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
            with st.spinner(f"Analyzing input {idx + 1}..."):
                try:
                    img_array, img_pil = preprocess_image(item)
                    pred_mask = tflite_predict(interpreter, img_array)
                    display_prediction(img_pil, pred_mask)
                    get_combined_download_links(img_pil, pred_mask, idx)
                except Exception as e:
                    st.error(f"❌ Error with input {idx + 1}: {e}")
    st.markdown('</div>', unsafe_allow_html=True)

# =============================
# 🎓 FOOTER
# =============================
# (The footer markdown remains the same)
st.markdown("""
<style>
.booking-style-footer { /* ... styles ... */ }
</style>
<div class="booking-style-footer">
    <!-- ... footer content ... -->
</div>
""", unsafe_allow_html=True)
