import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ===============================
# Page config
# ===============================
st.set_page_config(
    page_title="Klasifikasi Kucing vs Anjing",
    page_icon="🐱",
    layout="centered"
)

# ===============================
# Custom CSS
# ===============================
st.markdown("""
<style>
.main {
    background-color: #f7f9fc;
}
.title {
    font-size: 42px;
    font-weight: bold;
    text-align: center;
}
.subtitle {
    font-size: 18px;
    text-align: center;
    color: #555;
}
.pred-box {
    padding: 5px;
    border-radius: 12px;
    text-align: center;
    margin-top: 20px;
}
.cat {
    background-color: #e3f2fd;
    color: #0d47a1;
}
.dog {
    background-color: #fce4ec;
    color: #880e4f;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# Load model (SAFE)
# ===============================


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "cats_vs_dogs_model.keras",
        compile=False
    )


model = load_model()

# ===============================
# Title
# ===============================
st.markdown("<div class='title'>Kucing vs Anjing</div>",
            unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload gambar kucing atau anjing dan biarkan AI menebak</div>",
            unsafe_allow_html=True)
st.markdown("---")

# ===============================
# Upload image
# ===============================
uploaded_file = st.file_uploader(
    "Upload gambar (jpg / png)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.image(image, caption="Gambar yang diupload", width=400)

    # ===============================
    # Preprocess
    # ===============================
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # ===============================
    # Prediction
    # ===============================
    with st.spinner("Menganalisis gambar..."):
        pred = model.predict(img_array)[0][0]

    cat_conf = 1 - pred
    dog_conf = pred

    # ===============================
    # Result UI
    # ===============================
    if pred > 0.5:
        st.markdown(
            f"""
            <div class='pred-box dog'>
                <h2>&nbsp;&nbsp;&nbsp;Anjing!</h2>
                <p>Confidence: <b>{dog_conf:.2%}</b></p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div class='pred-box cat'>
                <h2>&nbsp;&nbsp;Kucing!</h2>
                <p>Confidence: <b>{cat_conf:.2%}</b></p>
            </div>
            """,
            unsafe_allow_html=True
        )

    # ===============================
    # Confidence bars
    # ===============================
    st.markdown("<h3>Confidence Level</h3>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1])
    with col1:
        st.progress(float(cat_conf))
        st.caption(f"Kucing: {cat_conf:.2%}")

    with col2:
        st.progress(float(dog_conf))
        st.caption(f"Anjing: {dog_conf:.2%}")

else:
    st.info("Silakan upload gambar kucing atau anjing")

# ===============================
# Footer
# ===============================
st.markdown("---")
st.caption("Model: MobileNetV2 + Hyperparameter Tuning | TensorFlow")
