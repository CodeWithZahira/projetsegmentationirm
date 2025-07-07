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
# 🖼️ BACKGROUND CSS PER SECTION
# =============================
section_styles = """
<style>
#section1 {
  background: url('https://wallpaperbat.com/img/6790-robotics-wallpaper.jpg') no-repeat center center;
  background-size: cover;
  padding: 500px;
  border-radius: 5px;
  color: white;
}
#section2 {
  background: url('https://cdn.pixabay.com/photo/2024/01/09/03/24/ai-generated-8496704_1280.jpg') no-repeat center center;
  background-size: cover;
  padding: 500px;
  border-radius: 5px;
  color: white;
}
#section3 {
  background: url('https://tse1.mm.bing.net/th/id/OIP.Q5jvjjsGVByg4_mzDdjAGAHaEK?pid=Api&P=0&h=180') no-repeat center center;
  background-size: cover;
  padding: 500px;
  border-radius: 5px;
  color: white;
}
</style>
"""
st.markdown(section_styles, unsafe_allow_html=True)

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
with st.container():
    st.markdown('<div id="section1">', unsafe_allow_html=True)
    st.markdown("""
        <h1 style='text-align: center;'>🧠 NeuroSeg: Brain MRI Segmentation</h1>
        <h4 style='text-align: center;'>A deep learning-powered assistant for brain MRI image segmentation</h4>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# =============================
# 📥 SECTION 2: Upload + Predict
# =============================
with st.container():
    st.markdown('<div id="section2">', unsafe_allow_html=True)
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

    st.markdown('</div>', unsafe_allow_html=True)

# =============================
# 📚 SECTION 3: About
# =============================
with st.container():
    st.markdown('<div id="section3">', unsafe_allow_html=True)
    st.subheader("📚 À propos")
    st.markdown("""
    Cette application a été développée par **Zahira** dans le cadre de son projet de Master en Ingénierie Biomédicale. 
    Elle vise à assister les professionnels de santé dans la segmentation des IRM cérébrales à l'aide de l'intelligence artificielle.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# =============================
# 🔻 FOOTER
# =============================
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #cccccc;'>Made with ❤️ by Zahira | 2025</p>", unsafe_allow_html=True)
