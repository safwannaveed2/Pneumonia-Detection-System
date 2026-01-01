import streamlit as st
import numpy as np
import cv2
import tensorflow

st.set_page_config(
    page_title="Pneumonia Detection System",
    page_icon="🩺",
    layout="centered"
)


@st.cache_resource
def load_trained_model():
    return tensorflow.keras.models.load_model("pnuemonia_model.keras")

model = load_trained_model()


st.title("🩺 Pneumonia Detection System")
st.write(
    "Upload a chest X-ray image to check whether **Pneumonia** is present or **Not**."
)


uploaded_file = st.file_uploader(
    "Upload Chest X-ray Image",
    type=["jpg", "jpeg", "png"]
)


def preprocess_image(image):
    img = cv2.resize(image, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img


if uploaded_file is not None:
    file_bytes = np.asarray(
        bytearray(uploaded_file.read()), dtype=np.uint8
    )
    image = cv2.imdecode(file_bytes, 1)

    st.image(image, caption="Uploaded MRI", use_container_width=True)

    if st.button("Predict"):
        processed_img = preprocess_image(image)
        prediction = model.predict(processed_img)[0][0]

        if prediction > 0.5:
            st.error("🩺 Pneumonia Detected")
        else:
            st.success("✅ No Pneumonia Detected")