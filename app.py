import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import io
import base64

# =============================
# 🎨 PAGE CONFIG
# =============================
st.set_page_config(page_title="NeuroSeg", layout="wide")

# =============================
# 🖼️ BACKGROUND CSS + NAVBAR
# =============================
nav_style = """
    <style>
    .stApp {
        background-image: url('https://images.unsplash.com/photo-1581093588401-01059c8a207c');
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
    }
    .main {
        background-color: rgba(0, 0, 0, 0.65);
        padding: 2rem;
        border-radius: 15px;
    }
    h1, h2, h3, h4, h5, h6, p, label, .stButton>button {
        color: white !important;
    }
    .custom-navbar {
        background-color: rgba(0,0,0,0.4);
        padding: 1rem;
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        z-index: 9999;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .custom-navbar a {
        margin: 0 1.5rem;
        color: white;
        text-decoration: none;
        font-weight: bold;
        font-size: 1.1rem;
    }
    .custom-navbar a:hover {
        text-decoration: underline;
    }
    .logo {
        font-size: 1.3rem;
        font-weight: bold;
        color: white;
    }
    .spacer {
        height: 80px;
    }
    </style>
"""
st.markdown(nav_style, unsafe_allow_html=True)

st.markdown("""
<div class="custom-navbar">
    <div class="logo">NeuroSeg</div>
    <div>
        <a href="#header">Accueil</a>
        <a href="#predict">Prédire</a>
        <a href="#about">À propos</a>
    </div>
</div>
<div class="spacer"></div>
""", unsafe_allow_html=True)

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
# 🧠 SECTION 1: HEADER
# =============================
st.markdown("<div class='main' id='header'>", unsafe_allow_html=True)
st.markdown("""
    <h1 style='text-align: center;'>🧠 NeuroSeg: Brain MRI Segmentation</h1>
    <h4 style='text-align: center;'>A deep learning-powered assistant for brain MRI image segmentation</h4>
""", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# =============================
# 📥 SECTION 2: Upload + Predict
# =============================
st.markdown("<div class='main' id='predict'>", unsafe_allow_html=True)
st.subheader("📥 Charger le Modèle TFLite")

model_file = st.file_uploader("Téléversez votre modèle (.tflite)", type=["tflite"])
if model_file is not None:
    try:
        tflite_model = model_file.read()
        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        st.success("✅ Modèle TFLite chargé avec succès.")
        model_loaded = True
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du modèle: {e}")
        model_loaded = False
else:
    model_loaded = False

st.subheader("🖼️ Téléverser une Image IRM")

image_file = st.file_uploader("Chargez une image IRM (PNG, JPG, TIF)", type=["png", "jpg", "jpeg", "tif", "tiff"])
if image_file is not None and model_loaded:
    img_array, img_pil = preprocess_image(image_file)
    st.image(img_pil, caption="Image originale", use_column_width=True)

    if st.button("🧠 Lancer la prédiction"):
        pred_mask = tflite_predict(interpreter, img_array)
        st.success("✅ Prédiction terminée !")
        display_prediction(img_pil, pred_mask)

st.markdown("</div>", unsafe_allow_html=True)

# =============================
# 📚 SECTION 3: About
# =============================
st.markdown("<div class='main' id='about'>", unsafe_allow_html=True)
st.subheader("📚 À propos")
st.markdown("""
Cette application a été développée par **Zahira** dans le cadre de son projet de Master en Ingénierie Biomédicale. 
Elle vise à assister les professionnels de santé dans la segmentation des IRM cérébrales à l'aide de l'intelligence artificielle.
""")
st.markdown("</div>", unsafe_allow_html=True)

# =============================
# 🔻 FOOTER
# =============================
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #ffffff;'>Made with ❤️ by Zahira | 2025</p>", unsafe_allow_html=True)
