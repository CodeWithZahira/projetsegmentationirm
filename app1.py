import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# ===== Dice coefficient & loss =====
def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (
        tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

# ===== Image Preprocessing =====
def preprocess_image(uploaded_file, target_size=(128, 128)):
    image = Image.open(uploaded_file).convert("L")  # Grayscale
    image = image.resize(target_size)
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=(0, -1))  # shape: (1, H, W, 1)
    return img_array, image

# ===== Prediction =====
def predict(model, image_array):
    prediction = model.predict(image_array)[0, :, :, 0]
    prediction = (prediction > 0.5).astype(np.uint8) * 255
    return prediction

# ===== Streamlit App =====
st.set_page_config(page_title="Segmentation IRM - UNet", layout="centered")
st.title("🧠 Application de Segmentation IRM - UNet")

st.markdown("### 1️⃣ Téléversez le modèle UNet (`.h5` ou `.keras`)")
model_file = st.file_uploader("📥 Modèle entraîné", type=["h5", "keras"])

if model_file is not None:
    try:
        model = tf.keras.models.load_model(model_file, compile=False)
        model.compile(optimizer='adam', loss=dice_loss, metrics=[dice_coef])
        st.success("✅ Modèle chargé avec succès.")
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du modèle: {e}")
        st.stop()

    st.markdown("### 2️⃣ Téléversez une image IRM à segmenter")
    image_file = st.file_uploader("📤 Image IRM (PNG ou JPG)", type=["png", "jpg", "jpeg"])

    if image_file is not None:
        img_array, img_pil = preprocess_image(image_file)
        st.image(img_pil, caption="🖼️ Image IRM originale", use_column_width=True)

        st.markdown("### 3️⃣ Résultat de la segmentation")
        prediction = predict(model, img_array)
        st.image(prediction, caption="🧾 Masque segmenté", use_column_width=True, clamp=True)
