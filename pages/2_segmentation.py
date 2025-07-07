import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import io

# ===== Image preprocessing =====
def preprocess_image(uploaded_file, target_size=(128, 128)):
    image = Image.open(uploaded_file).convert("L")
    image = image.resize(target_size)
    img_array = np.array(image) / 255.0
    img_array = img_array.astype(np.float32)
    img_array = np.expand_dims(img_array, axis=(0, -1))  # (1, H, W, 1)
    return img_array, image

# ===== TFLite prediction =====
def tflite_predict(interpreter, input_data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    prediction = output_data[0, :, :, 0]
    prediction = (prediction > 0.5).astype(np.uint8) * 255
    return prediction

# ===== Display side by side =====
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

# ===== Streamlit app =====
st.title("🧠 Segmentation IRM avec modèle TFLite")

model_file = st.file_uploader("📥 Téléversez le modèle TFLite (.tflite)", type=["tflite"])

if model_file is not None:
    try:
        tflite_model = model_file.read()
        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        st.success("✅ Modèle TFLite chargé.")
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du modèle: {e}")
        st.stop()

    image_file = st.file_uploader("📤 Téléversez une image IRM (PNG/JPG/TIF)", type=["png", "jpg", "jpeg", "tif", "tiff"])
    if image_file is not None:
        img_array, img_pil = preprocess_image(image_file)
        st.image(img_pil, caption="Image originale", use_column_width=True)

        if st.button("🧠 Lancer la prédiction"):
            pred_mask = tflite_predict(interpreter, img_array)
            display_prediction(img_pil, pred_mask)
