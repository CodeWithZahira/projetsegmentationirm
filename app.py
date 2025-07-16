import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import io

# =============================
# 🔧 PAGE CONFIG
# =============================
st.set_page_config(page_title="NeuroSeg Base", layout="centered")



st.markdown("""
<style>
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}
body {
  background-color: #000;
}
.codigo {
  position: absolute;
  top: 0;
  left: 50%;
  transform: translateX(-50%);
  background: #272822;
  color: #F8F8F2;
  font-family: 'Consolas', 'Monaco', monospace;
  font-size: 12px;
  padding: 20px;
  border-radius: 8px;
  white-space: pre-line;
  text-align: left;
  line-height: 1.6;
  max-height: 50%;
  width: 90%;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
}
.selector       { color: #F92672; }
.property       { color: #66D9EF; }
.value-number   { color: #AE81FF; }
.value-string   { color: #E6DB74; }
.value-color    { color: #A6E22E; }
.brace          { color: #F8F8F2; }
.comment        { color: #75715E; }
.function       { color: #A6E22E; }

.animation-container {
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
  height: 300px;
  overflow: hidden;
  background-color: #000;
  margin-bottom: 30px;
}

.box {
  width: 100px;
  height: 100px;
  background-color: #fff;
  margin: 0 10px;
}

.box:nth-child(1) {
  animation: box1 2s infinite;
}
.box:nth-child(2) {
  animation: box2 2s infinite;
}
.box:nth-child(3) {
  animation: box3 2s infinite;
}

@keyframes box1 {
  0%   { transform: translate(0, 0); opacity: 1; }
  50%  { transform: translate(50px, 0); opacity: 0.5; }
  100% { transform: translate(0, 0); opacity: 1; }
}

@keyframes box2 {
  0%   { transform: translate(0, 0); opacity: 1; }
  50%  { transform: translate(0, 50px); opacity: 0.5; }
  100% { transform: translate(0, 0); opacity: 1; }
}

@keyframes box3 {
  0%   { transform: translate(0, 0); opacity: 1; rotate: 0deg; }
  50%  { transform: translate(0, 0); opacity: 0.5; rotate: 360deg; }
  100% { transform: translate(0, 0); opacity: 1; rotate: 0deg; }
}
</style>
""", unsafe_allow_html=True)



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
# 🧠 APP TITLE
# =============================
st.title("🧠 NeuroSeg - Segmentation IRM")

st.markdown("Ce projet utilise un modèle TFLite pour segmenter les images IRM téléchargées.")

# =============================
# 📁 MODEL UPLOAD
# =============================
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
