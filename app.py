import streamlit as st
from PIL import Image

# Page configuration
st.set_page_config(page_title="MRI Segmentation", layout="centered")

# Apply custom CSS
st.markdown("""
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { background-color: #000; }
.animation-container {
  position: relative;
  display: flex;
  justify-content: center;
  align-items: center;
  height: 200px;
  margin: 30px 0;
}
.box {
  width: 80px;
  height: 80px;
  background-color: white;
  margin: 0 10px;
}
.box:nth-child(1) { animation: box1 2s infinite; }
.box:nth-child(2) { animation: box2 2s infinite; }
.box:nth-child(3) { animation: box3 2s infinite; }

@keyframes box1 {
  0% { transform: translate(0, 0); opacity: 1; }
  50% { transform: translate(50px, 0); opacity: 0.5; }
  100% { transform: translate(0, 0); opacity: 1; }
}
@keyframes box2 {
  0% { transform: translate(0, 0); opacity: 1; }
  50% { transform: translate(0, 50px); opacity: 0.5; }
  100% { transform: translate(0, 0); opacity: 1; }
}
@keyframes box3 {
  0% { transform: translate(0, 0); opacity: 1; rotate: 0deg; }
  50% { transform: translate(0, 0); opacity: 0.5; rotate: 360deg; }
  100% { transform: translate(0, 0); opacity: 1; rotate: 0deg; }
}
</style>
""", unsafe_allow_html=True)

# Title
st.title("🧠 MRI Brain Tumor Segmentation")

# Animation
st.markdown("""
<div class="animation-container">
  <div class="box"></div>
  <div class="box"></div>
  <div class="box"></div>
</div>
""", unsafe_allow_html=True)

# Image uploader
uploaded_file = st.file_uploader("Upload an MRI image (JPG or PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    st.success("Image uploaded successfully!")

    # Placeholder for model
    if st.button("Segment Image"):
        st.warning("🧠 Segmentation not yet implemented. You can integrate your model here.")

else:
    st.info("Please upload an image to start.")
