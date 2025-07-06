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

# -------- Custom dice functions --------
def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)
# -------- Download model from Drive --------
def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("📥 Téléchargement du modèle depuis Google Drive..."):
            gdown.download(id=MODEL_DRIVE_ID, output=MODEL_PATH, quiet=False)
        st.success("✅ Modèle téléchargé avec succès!")

# -------- Load U-Net model --------
def load_unet_model():
    model = load_model(MODEL_PATH, custom_objects={'dice_loss': dice_loss, 'dice_coef': dice_coef})
    return model

# -------- Preprocessing image --------
def preprocess_image(img: Image.Image):
    if img.mode != "L":
        img = img.convert("L")  # Convertir en niveaux de gris

    img = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, IMG_SIZE[0], IMG_SIZE[1], 1)
    return img_array
st.title("Brain MRI Segmentation App")
model = load_model(MODEL_PATH, custom_objects={'dice_loss': dice_loss, 'dice_coef': dice_coef})

im_height = 256
im_width = 256

file = st.file_uploader("Upload file", type=[
                            "csv", "png", "jpg"], accept_multiple_files=True)
if file:
    for i in file:
        st.header("Original Image:")
        st.image(i)
        content = i.getvalue()
        image = np.asarray(bytearray(content), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        img2 = cv2.resize(image, (im_height, im_width))
        img3 = img2/255
        img4 = img3[np.newaxis, :, :, :]
        if st.button("Predict Output:"):
            pred_img = model.predict(img4)
            st.header("Predicted Image:")
            st.image(pred_img)
        else:
            continue
