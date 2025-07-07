import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import io

# ============== PAGE CONFIG ===================
st.set_page_config(page_title="NeuroSeg - IRM App", layout="wide")

# ============== STYLES ===================
st.markdown("""
    <style>
    html, body, .stApp {
        height: 100%;
        margin: 0;
        font-family: 'Segoe UI', sans-serif;
        background: url('https://img.freepik.com/premium-photo/concept-art-human-brain-exploding-with-knowledge-creativity-generative-ai_438099-10972.jpg') no-repeat center center fixed;
        background-size: cover;
        color: white;
    }

    h1 {
        font-size: 3.5em;
        font-weight: bold;
        animation: fadeIn 2s ease-in-out;
    }

    h2, h3 {
        color: white;
    }

    .container {
        background-color: rgba(0, 0, 0, 0.6);
        padding: 60px 40px;
        border-radius: 15px;
        margin-top: 100px;
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

    .prediction-section {
        background-color: rgba(0, 0, 0, 0.6);
        padding: 30px;
        border-radius: 12px;
    }

    @keyframes fadeIn {
        0% {opacity: 0; transform: translateY(-20px);}
        100% {opacity: 1; transform: translateY(0);}
    }
    </style>
""", unsafe_allow_html=True)

# ============== UTILS ===================
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

# ============== MAIN UI ===================
with st.container():
    st.markdown("<h1 style='text-align: center;'>🧠 NeuroSeg</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>Une application IA pour la segmentation des IRM</h3>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    with st.container():
        st.markdown("<div class='prediction-section'>", unsafe_allow_html=True)
        st.header("📤 Upload et Prédiction")

        model_file = st.file_uploader("Téléversez votre modèle TFLite (.tflite)", type=["tflite"])
        model_loaded = False
        if model_file is not None:
            try:
                tflite_model = model_file.read()
                interpreter = tf.lite.Interpreter(model_content=tflite_model)
                interpreter.allocate_tensors()
                st.success("✅ Modèle chargé avec succès.")
                model_loaded = True
            except Exception as e:
                st.error(f"❌ Erreur: {e}")

        image_file = st.file_uploader("Téléversez une image IRM (PNG/JPG/TIF)", type=["png", "jpg", "jpeg", "tif", "tiff"])
        if image_file is not None and model_loaded:
            img_array, img_pil = preprocess_image(image_file)
            st.image(img_pil, caption="Image originale", use_column_width=True)

            if st.button("🧠 Lancer la prédiction", type="primary"):
                pred_mask = tflite_predict(interpreter, img_array)
                display_prediction(img_pil, pred_mask)
        st.markdown("</div>", unsafe_allow_html=True)

# ============== FOOTER ===================
st.markdown("""
<br><br><hr>
<p style='text-align: center;'>© 2025 NeuroSeg — by Zahira</p>
""", unsafe_allow_html=True)
