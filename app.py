import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import io
import requests
import base64

# =============================
# 🎨 PAGE CONFIG
# =============================
st.set_page_config(page_title="NeuroSeg", layout="wide")

# =============================
# 🖼️ BACKGROUND SETUP (using base64 image)
# =============================
def set_bg_from_url(image_url):
    response = requests.get(image_url)
    encoded_string = base64.b64encode(response.content).decode()
    st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded_string}");
            background-size: cover;
            background-attachment: fixed;
        }}
        </style>
    """, unsafe_allow_html=True)

bg_image = "https://png.pngtree.com/png-vector/20240201/ourmid/pngtree-ai-generative-human-brain-illustration-png-image_11529766.png"
set_bg_from_url(bg_image)

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
# 🧠 APP STRUCTURE (scroll layout)
# =============================

# ----- Header -----
st.markdown("""
    <h1 style='text-align: center; color: #ffffff;'>🧠 NeuroSeg: Brain MRI Segmentation</h1>
    <h4 style='text-align: center; color: #dddddd;'>A deep learning-powered assistant for brain MRI image segmentation</h4>
""", unsafe_allow_html=True)

st.markdown("---")

# ----- Section: Upload model -----
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

st.markdown("---")

# ----- Section: Image upload & prediction -----
st.subheader("🖼️ Téléverser une Image IRM")

image_file = st.file_uploader("Chargez une image IRM (PNG, JPG, TIF)", type=["png", "jpg", "jpeg", "tif", "tiff"])
if image_file is not None and model_loaded:
    img_array, img_pil = preprocess_image(image_file)
    st.image(img_pil, caption="Image originale", use_column_width=True)

    if st.button("🧠 Lancer la prédiction"):
        pred_mask = tflite_predict(interpreter, img_array)
        st.success("✅ Prédiction terminée !")
        display_prediction(img_pil, pred_mask)

st.markdown("---")

# ----- About section -----
st.subheader("📚 À propos")
st.markdown("""
Cette application a été développée par **Zahira** dans le cadre de son projet de Master en Ingénierie Biomédicale. 
Elle vise à assister les professionnels de santé dans la segmentation des IRM cérébrales à l'aide de l'intelligence artificielle.
""")

# ----- Footer -----
st.markdown("---")
st.markdown("<p style='text-align: center; color: #cccccc;'>Made with ❤️ by Zahira | 2025</p>", unsafe_allow_html=True)
