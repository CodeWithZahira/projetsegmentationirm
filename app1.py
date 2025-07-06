import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import io
import requests
import tempfile

# ========== CONFIGURATION ==========
st.set_page_config(page_title="🧠 NeuroSeg - IRM Segmentation", page_icon="🧠", layout="wide")
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🧠 NeuroSeg</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Segmentation automatique des IRM cérébrales avec Deep Learning</h4>", unsafe_allow_html=True)
st.markdown("---")

# ========== TELECHARGER LE MODELE DEPUIS GOOGLE DRIVE ==========
@st.cache_resource
def load_tflite_model_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Erreur de téléchargement du modèle: {e}")
        return None

    with tempfile.NamedTemporaryFile(delete=False, suffix=".tflite") as tmp_file:
        tmp_file.write(response.content)
        model_path = tmp_file.name

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

https://drive.google.com/file/d/16sxehD5rCA9uVovFIsgqMQDiL_5fAT8u/view?usp=sharing
MODEL_ID = "16sxehD5rCA9uVovFIsgqMQDiL_5fAT8u"
MODEL_URL = f"https://drive.google.com/uc?export=download&id={16sxehD5rCA9uVovFIsgqMQDiL_5fAT8u}"

interpreter = load_tflite_model_from_url(MODEL_URL)

if interpreter:
    st.success("✅ Modèle chargé automatiquement depuis Google Drive.")
else:
    st.stop()

# ========== PRETRAITEMENT DE L’IMAGE ==========
def preprocess_image(uploaded_file, target_size=(128, 128)):
    image = Image.open(uploaded_file).convert("L")
    image = image.resize(target_size)
    img_array = np.array(image) / 255.0
    img_array = img_array.astype(np.float32)
    img_array = np.expand_dims(img_array, axis=(0, -1))  # (1, H, W, 1)
    return img_array, image

# ========== PREDICTION ==========
def tflite_predict(interpreter, input_data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    prediction = output_data[0, :, :, 0]
    binary_mask = (prediction > 0.5).astype(np.uint8) * 255
    return prediction, binary_mask

# ========== AFFICHAGE DES RESULTATS ==========
def display_results(original_img, raw_mask, bin_mask):
    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    axs[0].imshow(original_img, cmap="gray")
    axs[0].set_title("🧠 Image Originale")
    axs[0].axis("off")

    axs[1].imshow(raw_mask, cmap="gray")
    axs[1].set_title("🧪 Masque Brut")
    axs[1].axis("off")

    axs[2].imshow(bin_mask, cmap="gray")
    axs[2].set_title("✅ Masque Final")
    axs[2].axis("off")

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    st.image(buf)

# ========== INTERFACE UTILISATEUR ==========
st.markdown("### 📤 Téléversez une image IRM (PNG / JPG / TIF)")
image_file = st.file_uploader("Choisissez une image IRM", type=["png", "jpg", "jpeg", "tif", "tiff"])

if image_file is not None:
    input_data, img_pil = preprocess_image(image_file)
    st.image(img_pil, caption="🖼️ Image originale", use_column_width=True)

    if st.button("🔍 Lancer la segmentation"):
        with st.spinner("⏳ Prédiction en cours..."):
            raw_mask, bin_mask = tflite_predict(interpreter, input_data)
            display_results(img_pil, raw_mask, bin_mask)
