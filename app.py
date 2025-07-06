import os
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
import gdown

# -------- Constants --------
MODEL_DRIVE_ID = "1ti_-AtZzOHTVzHAHnQJyJuNzfu4TD7IB"
MODEL_PATH = "unet_model_finetuned.h5"
IMG_SIZE = (128, 128)



# -------- Download model from Drive --------
def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("📥 Téléchargement du modèle depuis Google Drive..."):
            gdown.download(id=MODEL_DRIVE_ID, output=MODEL_PATH, quiet=False)
        st.success("✅ Modèle téléchargé avec succès!")

# -------- Load U-Net model --------

def load_unet_model_finetuned():
    with st.spinner("🔄 Chargement du modèle..."):
        model = load_model(MODEL_PATH, custom_objects={
            'dice_loss': dice_loss,
            'dice_coef': dice_coef
        }, compile=False)
        model.compile(optimizer='adam', loss=dice_loss, metrics=[dice_coef])
        return model

# ثم:
model = load_unet_model_finetuned()

# -------- Preprocessing image --------
def preprocess_image(img: Image.Image):
    if img.mode != "L":
        img = img.convert("L")  # Convertir en niveaux de gris

    img = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, IMG_SIZE[0], IMG_SIZE[1], 1)
    return img_array

# -------- Début de l'app --------
st.set_page_config(page_title="IRM Brain Segmentation", layout="centered")
st.title("🧠 IRM Brain Segmentation avec U-Net")
st.markdown("Ce projet utilise un modèle U-Net pour segmenter les IRMs cérébrales.")

# -------- Télécharger et charger le modèle --------
download_model()
model = load_unet_model_finetuned()

# -------- Upload de l'image --------
uploaded_file = st.file_uploader("📤 Choisissez une image IRM (.png)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Charger et afficher l'image
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼️ Image chargée", use_column_width=True)

    # Bouton de prédiction
    if st.button("🔍 Segmenter"):
        with st.spinner("⏳ Prédiction en cours..."):
            input_img = preprocess_image(image)
            prediction = model.predict(input_img)[0, :, :, 0]

            # Seuillage simple pour obtenir un masque binaire
            mask = (prediction > 0.5).astype(np.uint8) * 255
            mask_img = Image.fromarray(mask)

        # Affichage
        st.subheader("🩻 Masque segmenté :")
        st.image(mask_img, use_column_width=True)

        # Option de téléchargement
        st.download_button("📥 Télécharger le masque", data=mask_img.tobytes(),
                           file_name="mask.png", mime="image/png")
