import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

image_path = "/tmp/test_uploaded_leaf.jpg"
model_path = "leaf_detector_augmented.keras"

model = tf.keras.models.load_model(model_path, compile=False)

image = tf.keras.utils.load_img(
    image_path,
    target_size=(160, 160)
)

array = tf.keras.utils.img_to_array(image)
array = preprocess_input(array)
array = np.expand_dims(array, axis=0)

score = float(model.predict(array, verbose=0)[0][0])
print(f"Plant detector score: {score * 100:.1f}%")
