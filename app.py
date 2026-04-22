import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from utils.pdf_handler import pdf_to_image
from utils.preprocess import preprocess_for_model

st.set_page_config(page_title="Digital Handwritten Recognition", page_icon="✍️", layout="centered")

st.title("✍️ Digital Hand Written Recognition")
st.markdown("**Upload a photo or PDF** — I'll tell you if it's **handwritten** or **printed**.")

# Load model once
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/handwritten_classifier.h5")

model = load_model()

# File uploader
uploaded_file = st.file_uploader("Choose image (JPG/PNG) or PDF", type=["jpg", "jpeg", "png", "pdf"])

if uploaded_file is not None:
    file_type = uploaded_file.type
    
    if "pdf" in file_type:
        st.info("Processing PDF (first page only)...")
        pdf_bytes = uploaded_file.read()
        image = pdf_to_image(pdf_bytes)
    else:
        image = Image.open(uploaded_file).convert("RGB")
    
    # Show image
    st.image(image, caption="Uploaded Document", use_column_width=True)
    
    # Preprocess & Predict
    with st.spinner("Scanning document..."):
        processed = preprocess_for_model(image)
        prediction = model.predict(processed, verbose=0)[0][0]
    
    # Result
    if prediction > 0.5:
        label = "🖨️ **Printed (Digital)**"
        confidence = prediction * 100
    else:
        label = "✍️ **Handwritten**"
        confidence = (1 - prediction) * 100
    
    st.success(label)
    st.metric("Confidence", f"{confidence:.1f}%")
    
    # Extra info
    st.caption("Model trained on real handwritten characters + synthetic printed text.")
