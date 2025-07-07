import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import io

# =============================
# 🎨 PAGE CONFIG
# =============================
st.set_page_config(page_title="NeuroSeg", layout="wide")

# =============================
# 🔧 UTILITIES
# =============================
def preprocess_image(uploaded_file, target_size=(128, 128)):
    image = Image.open(uploaded_file).convert("L")
    image = image.resize(target_size)
    img_array = np.array(image) / 255.0
    img_array = img_array.astype(np.float32)
    img_array = np.expand_dims(img_array, axis=(0, -1))  # (1, H, W, 1)
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
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].imshow(image_pil, cmap="gray")
    axs[0].set_title("Image Originale")
    axs[0].axis("off")

    axs[1].imshow(mask, cmap="gray")
    axs[1].set_title("Masque Prédit")
    axs[1].axis("off")

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    st.image(buf)

# =============================
# 💅 STYLES AND ANIMATION
# =============================
st.markdown("""
    <style>
    html, body, .stApp {
        margin: 0;
        padding: 0;
        font-family: 'Segoe UI', sans-serif;
        scroll-behavior: smooth;
    }
    .navbar {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        background-color: rgba(0, 0, 0, 0.8);
        padding: 1rem 2rem;
        display: flex;
        justify-content: space-between;
        z-index: 9999;
    }
    .navbar a {
        color: white;
        margin: 0 15px;
        text-decoration: none;
        font-weight: bold;
        font-size: 16px;
    }
    .navbar a:hover {
        text-decoration: underline;
    }
    .logo {
        font-size: 20px;
        font-weight: bold;
        color: #fff;
    }
    .section {
        height: 100vh;
        padding: 100px 100px;
        color: white;
        text-align: center;
        display: flex;
        flex-direction: column;
        justify-content: center;
        animation: fadeIn 2s ease-in-out;
    }
    #home {
        background: url('https://images.unsplash.com/photo-1603791440384-56cd371ee9a7') no-repeat center center;
        background-size: cover;
    }
    #predict {
        background: url('https://images.unsplash.com/photo-1581093588401-01059c8a207c') no-repeat center center;
        background-size: cover;
    }
    #about {
        background: url('https://images.unsplash.com/photo-1530026405186-ed1f139313d7') no-repeat center center;
        background-size: cover;
    }
    h1.animated-title {
        font-size: 3.5em;
        animation: slideDown 2s ease-out;
    }
    @keyframes slideDown {
        0% { opacity: 0; transform: translateY(-50px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    @keyframes fadeIn {
        from {opacity: 0;}
        to {opacity: 1;}
    }
    .btn-main {
        background-color: #FF4B4B;
        padding: 10px 25px;
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 25px;
        font-size: 16px;
        cursor: pointer;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# =============================
# 🔝 NAVBAR
# =============================
st.markdown("""
<div class="navbar">
  <div class="logo">Université XYZ</div>
  <div>
    <a href="#home">Accueil</a>
    <a href="#predict">Prédiction</a>
    <a href="#about">Contact</a>
  </div>
</div>
""", unsafe_allow_html=True)

# =============================
# 🏠 SECTION: HOME
# =============================
st.markdown("""
<section id="home" class="section">
  <h1 class="animated-title">Bienvenue sur NeuroSeg</h1>
  <h3>Application de segmentation IRM basée sur l'IA</h3>
  <p>Crée par Zahira - Université XYZ</p>
  <a href="#predict"><button class="btn-main">Commencer</button></a>
</section>
""", unsafe_allow_html=True)

# =============================
# 🧠 SECTION: PREDICTION
# =============================
st.markdown("""
<section id="predict" class="section">
  <h2>🧠 Téléversez et prédisez</h2>
""", unsafe_allow_html=True)

model_file = st.file_uploader("Modèle TFLite (.tflite)", type=["tflite"])
model_loaded = False
if model_file is not None:
    try:
        tflite_model = model_file.read()
        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        st.success("✅ Modèle chargé")
        model_loaded = True
    except Exception as e:
        st.error(f"❌ Erreur: {e}")

image_file = st.file_uploader("Image IRM (PNG/JPG/TIF)", type=["png", "jpg", "jpeg", "tif", "tiff"])
if image_file and model_loaded:
    img_array, img_pil = preprocess_image(image_file)
    st.image(img_pil, caption="Image originale", use_column_width=True)
    if st.button("🔍 Prédire"):
        pred_mask = tflite_predict(interpreter, img_array)
        display_prediction(img_pil, pred_mask)

st.markdown("""
</section>
""", unsafe_allow_html=True)

# =============================
# ℹ️ SECTION: ABOUT / CONTACT
# =============================
st.markdown("""
<section id="about" class="section">
  <h2>📞 Contact</h2>
  <p>Développé dans le cadre du Master en Ingénierie Biomédicale - Université XYZ</p>
  <p>Email: zahira.etudiante@xyz.ac.ma</p>
</section>
""", unsafe_allow_html=True)

# =============================
# 🔻 FOOTER
# =============================
st.markdown("""
<hr>
<p style='text-align: center; color: white;'>© 2025 NeuroSeg. Made by Zahira.</p>
""", unsafe_allow_html=True)
