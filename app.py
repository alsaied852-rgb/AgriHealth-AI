import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

st.set_page_config(
    page_title="AgriHealth AI",
    page_icon="🌿",
    layout="centered"
)

CLASS_NAMES = [
    "Healthy",
    "Early Blight",
    "Late Blight",
    "Leaf spot",
    "Powdery mildew",
]

@st.cache_resource
def load_disease_model():
    # The model was exported from Teachable Machine in Keras (.h5) format.
    return load_model("keras_model.h5", compile=False)

st.title("🌿 AgriHealth AI")
st.subheader("Plant Disease Detection")
st.write(
    "Upload a clear photo of a plant leaf. The trained model will predict "
    "one of the five classes below."
)

st.caption(
    "Classes: 0 Healthy · 1 Early Blight · 2 Late Blight · "
    "3 Leaf spot · 4 Powdery mildew"
)

uploaded_file = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded image", use_container_width=True)

        if st.button("Analyze Image", type="primary"):
            with st.spinner("Analyzing image..."):
                model = load_disease_model()

                # Teachable Machine Keras image models expect 224x224 RGB
                # pixels normalized to the range [-1, 1].
                resized_image = image.resize((224, 224))
                image_array = np.asarray(resized_image, dtype=np.float32)
                image_array = (image_array / 127.5) - 1.0
                image_array = np.expand_dims(image_array, axis=0)

                predictions = model.predict(image_array, verbose=0)[0]
                predicted_index = int(np.argmax(predictions))
                predicted_label = CLASS_NAMES[predicted_index]
                confidence = float(predictions[predicted_index]) * 100

            st.markdown("### Prediction Result")
            st.success(f"Predicted class: **{predicted_label}**")
            st.metric("Confidence", f"{confidence:.2f}%")
            st.write("Confidence scores for all classes:")

            for index, class_name in enumerate(CLASS_NAMES):
                score = float(predictions[index])
                st.write(f"{index} — {class_name}: {score * 100:.2f}%")
                st.progress(max(0.0, min(1.0, score)))

            st.warning(
                "This is a model prediction, not a confirmed agricultural diagnosis. "
                "Confidence is not the same as overall model accuracy. Results may "
                "be unreliable for unclear images, other plant species, or conditions "
                "not represented in the training data."
            )
    except Exception as error:
        st.error(
            "The image could not be analyzed. Check that keras_model.h5 and "
            "labels.txt are in the same folder as app.py and that the required "
            "packages are installed."
        )
        st.exception(error)
else:
    st.info("Choose a JPG, JPEG, or PNG image to begin.")
