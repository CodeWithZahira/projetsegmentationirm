import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import io
import streamlit.components.v1 as com

# =============================
# 🔧 PAGE CONFIG
# =============================
st.set_page_config(page_title="NeuroSeg Base")  

# =============================
# 💬 Bienvenue + Animation
# =============================
with st.container():
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(
            "<h2 style='color:#fff; font-family:Palace_Script_MT;'>👋 Bienvenue sur NeuroSeg</h2>"
            "<p style='color:#ccc; font-size:18px;'>Téléversez votre modèle et image IRM pour voir la magie de la segmentation en action !</p>",
            unsafe_allow_html=True
        )
    with col2:
        com.iframe(
            "https://lottie.host/embed/f18f3de4-bd26-4c40-a8e8-4d57c67b5142/sQWEZtzUW3.lottie",
            height=380 , width = 300
        )

# =============================
# 📦 UTILITIES
# =============================
def preprocess_image(uploaded_file, target_size=(128, 128)):
    image = Image.open(uploaded_file).convert("L")
    image = image.resize(target_size)
    img_array = np.array(image) / 255.0
    img_array = img_array.astype(np.float32)
    img_array = np.expand_dims(img_array, axis=(0, -1))  # Shape: (1, H, W, 1)
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
# 📁 MODEL UPLOAD
# =============================
st.markdown("---")
model_file = st.file_uploader("📁 Importer le modèle (.tflite)", type=["tflite"])
model_loaded = False
if model_file is not None:
    try:
        tflite_model = model_file.read()
        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        st.success("✅ Modèle chargé avec succès.")
        model_loaded = True
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du modèle : {e}")

# =============================
# 🖼️ IMAGE UPLOAD & PREDICTION
# =============================
image_file = st.file_uploader("🖼️ Importer une image IRM (PNG/JPG/TIF)", type=["png", "jpg", "jpeg", "tif", "tiff"])
if image_file and model_loaded:
    img_array, img_pil = preprocess_image(image_file)
    st.image(img_pil, caption="Image originale", use_column_width=True)
    if st.button("🔍 Prédire"):
        pred_mask = tflite_predict(interpreter, img_array)
        display_prediction(img_pil, pred_mask)
