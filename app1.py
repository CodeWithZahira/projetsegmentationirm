import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import gdown
import os

MODEL_PATH = "unet_model.h5"
FILE_ID = "1ti_-AtZzOHTVzHAHnQJyJuNzfu4TD7IB"

def download_model():
    if not os.path.exists(MODEL_PATH):
        st.info("📥 Téléchargement du modèle...")
        gdown.download(id=FILE_ID, output=MODEL_PATH, quiet=False)

@st.cache_resource
def load_model():
    download_model()
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects={'dice_loss': dice_loss, 'dice_coef': dice_coef})
    return model

def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (
        tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def preprocess_image(uploaded_file, target_size=(128, 128)):
    image = Image.open(uploaded_file).convert("L")  # Grayscale
    image = image.resize(target_size)
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=(0, -1))  # (1, H, W, 1)
    return img_array, image

def predict(model, image_array):
    prediction = model.predict(image_array)[0, :, :, 0]
    prediction = (prediction > 0.5).astype(np.uint8) * 255
    return prediction

# Streamlit
st.set_page_config(page_title="Segmentation IRM Cerveau", layout="centered")
st.title("🧠 Application de Segmentation IRM - UNet")

uploaded_file = st.file_uploader("📤 Téléversez une image IRM (format PNG ou JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    model = load_model()

    st.subheader("📸 Image originale")
    image_array, image_display = preprocess_image(uploaded_file)
    st.image(image_display, caption="Image IRM", use_column_width=True)

    st.subheader("🔍 Prédiction de la segmentation...")
    mask = predict(model, image_array)

    st.subheader("🧾 Masque segmenté")
    st.image(mask, caption="Masque UNet", use_column_width=True, clamp=True)
