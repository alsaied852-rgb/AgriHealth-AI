
import streamlit as st
import numpy as np
from PIL import Image, UnidentifiedImageError
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
    return load_model("keras_model.h5", compile=False)

st.title("🌿 AgriHealth AI")
st.subheader("Plant Disease Detection")

st.write(
    "Upload a clear photo showing a single plant leaf. "
    "The model predicts one of five classes."
)

st.info(
    "Important: Upload a plant-leaf image only. This model does not "
    "verify whether an image contains a plant and may give incorrect "
    "predictions for objects, people, or unrelated images."
)

st.caption(
    "Classes: Healthy · Early Blight · Late Blight · "
    "Leaf spot · Powdery mildew"
)

uploaded_file = st.file_uploader(
    "Upload a plant leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(
            image,
            caption="Uploaded image — check that it clearly shows a plant leaf",
            use_container_width=True
        )

        st.warning(
            "Before analyzing, confirm that the image clearly shows "
            "a plant leaf. This confirmation is not an automatic image check."
        )

        user_confirms_leaf = st.checkbox(
            "I confirm this image clearly shows a plant leaf."
        )

        if st.button("Analyze Image", type="primary"):
            if not user_confirms_leaf:
                st.error(
                    "Please confirm that the image shows a plant leaf "
                    "before continuing."
                )
            else:
                with st.spinner("Analyzing image..."):
                    model = load_disease_model()

                    resized_image = image.resize((224, 224))
                    image_array = np.asarray(
                        resized_image, dtype=np.float32
                    )
                    image_array = (image_array / 127.5) - 1.0
                    image_array = np.expand_dims(image_array, axis=0)

                    predictions = model.predict(
                        image_array, verbose=0
                    )[0]

                    predicted_index = int(np.argmax(predictions))
                    predicted_label = CLASS_NAMES[predicted_index]
                    confidence = float(predictions[predicted_index]) * 100

                st.markdown("### Prediction Result")
                st.success(f"Predicted class: **{predicted_label}**")
                st.metric("Model score", f"{confidence:.2f}%")

                st.write("Scores for all classes:")
                for index, class_name in enumerate(CLASS_NAMES):
                    score = float(predictions[index])
                    st.write(
                        f"{class_name}: {score * 100:.2f}%"
                    )
                    st.progress(max(0.0, min(1.0, score)))

                st.warning(
                    "This result is a model prediction, not a confirmed "
                    "agricultural diagnosis. The model score is not the "
                    "probability that the result is correct or a measure "
                    "of overall model accuracy. The model may be confidently "
                    "wrong, especially for non-plant images, other plant "
                    "species, poor-quality images, or conditions outside "
                    "its training data. Do not use this result alone to "
                    "choose pesticides or treatment."
                )

    except (UnidentifiedImageError, OSError):
        st.error(
            "The uploaded file could not be read as an image. "
            "Please upload a valid JPG, JPEG, or PNG file."
        )
    except Exception as error:
        st.error(
            "The image could not be analyzed. Check that keras_model.h5 "
            "is available and that the required packages are installed."
        )
        st.exception(error)

else:
    st.info("Choose a JPG, JPEG, or PNG image to begin.")
