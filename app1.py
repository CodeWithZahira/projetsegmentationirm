import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

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
    # افترض أن الناتج هو (1, H, W, 1)
    prediction = output_data[0, :, :, 0]
    prediction = (prediction > 0.5).astype(np.uint8) * 255
    return prediction

# ===== Streamlit app =====
st.title("🧠 Segmentation IRM avec modèle TFLite")

model_file = st.file_uploader("📥 Téléversez le modèle TFLite (.tflite)", type=["tflite"])

if model_file is not None:
    # نقرأ الملف في الذاكرة
    tflite_model = model_file.read()
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    st.success("✅ Modèle TFLite chargé.")

    image_file = st.file_uploader("📤 Téléversez une image IRM (PNG/JPG)", type=["png", "jpg", "jpeg"])
    if image_file is not None:
        img_array, img_pil = preprocess_image(image_file)
        st.image(img_pil, caption="Image originale", use_column_width=True)

        pred_mask = tflite_predict(interpreter, img_array)
        st.image(pred_mask, caption="Masque segmenté", use_column_width=True, clamp=True)
