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
# 💅 STYLES
# =============================
st.markdown("""
    <style>
    html, body, .stApp {
        font-family: 'Segoe UI', sans-serif;
        scroll-behavior: smooth;
    }
    .section {
        padding: 100px 30px;
        color: white;
        text-align: center;
    }
    .section h1, .section h2, .section h3, .section p {
        color: white;
    }
    #hero {
        background: url('https://images.unsplash.com/photo-1581093588401-01059c8a207c') no-repeat center center;
        background-size: cover;
    }
    #predict-section {
        background: url('https://images.unsplash.com/photo-1603791440384-56cd371ee9a7') no-repeat center center;
        background-size: cover;
    }
    #about-section {
        background: #111;
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
    .btn-main {
        background-color: #FF4B4B;
        padding: 10px 25px;
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 25px;
        font-size: 16px;
        cursor: pointer;
    }
    .spacer { height: 80px; }
    </style>
""", unsafe_allow_html=True)

# =============================
# 🔝 NAVBAR
# =============================
st.markdown("""
<div class="navbar">
  <div class="logo">NeuroSeg</div>
  <div>
    <a href="#hero">Accueil</a>
    <a href="#predict-section">Prédire</a>
    <a href="#about-section">À propos</a>
  </div>
</div>
<div class="spacer"></div>
""", unsafe_allow_html=True)

# =============================
# 🧠 SECTION 1: HERO
# =============================
st.markdown("""
<div id="hero" class="section">
  <h1>🧠 Bienvenue sur NeuroSeg</h1>
  <h3>Une application intelligente pour la segmentation d’IRM cérébrales</h3>
  <p>Alimentée par l'apprentissage profond</p>
  <br>
  <a href="#predict-section"><button class="btn-main">Commencer</button></a>
</div>
""", unsafe_allow_html=True)

# =============================
# 📥 SECTION 2: PREDICT
# =============================
st.markdown("""
<div id="predict-section" class="section">
""", unsafe_allow_html=True)

st.header("🧠 Lancer une prédiction")
model_file = st.file_uploader("📥 Téléversez le modèle TFLite", type=["tflite"])
if model_file is not None:
    try:
        tflite_model = model_file.read()
        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        st.success("✅ Modèle TFLite chargé avec succès.")
        model_loaded = True
    except Exception as e:
        st.error(f"❌ Erreur de chargement: {e}")
        model_loaded = False
else:
    model_loaded = False

image_file = st.file_uploader("📷 Téléversez une image IRM (PNG/JPG/TIF)", type=["png", "jpg", "jpeg", "tif", "tiff"])
if image_file is not None and model_loaded:
    img_array, img_pil = preprocess_image(image_file)
    st.image(img_pil, caption="Image originale", use_column_width=True)

    if st.button("🔍 Prédire la segmentation"):
        pred_mask = tflite_predict(interpreter, img_array)
        display_prediction(img_pil, pred_mask)

st.markdown("""</div>""", unsafe_allow_html=True)

# =============================
# ℹ️ SECTION 3: ABOUT
# =============================
st.markdown("""
<div id="about-section" class="section">
  <h2>📚 À propos de cette application</h2>
  <p>
    Cette application a été développée par <strong>Zahira</strong> dans le cadre de son projet de Master en Ingénierie Biomédicale.<br>
    Elle vise à assister les professionnels de santé dans la détection des tumeurs cérébrales via l'IA.
  </p>
</div>
""", unsafe_allow_html=True)

# =============================
# 🔻 FOOTER
# =============================
st.markdown("""
<hr>
<p style='text-align: center; color: white;'>© 2025 NeuroSeg. Made with ❤️ by Zahira.</p>
""", unsafe_allow_html=True)
