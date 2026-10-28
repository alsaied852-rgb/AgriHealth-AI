import os
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("leaf_detector.keras")
folders = {
    "non_plant": "data/non_plant",
    "plant": "data/plant_leaf"
}

for expected, folder in folders.items():
    print(f"\nExpected class: {expected}")
    files = [
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ][:5]

    for filename in files:
        path = os.path.join(folder, filename)
        img = tf.keras.utils.load_img(path, target_size=(160, 160))
        arr = tf.keras.utils.img_to_array(img)
        arr = np.expand_dims(arr, axis=0)
        score = float(model.predict(arr, verbose=0)[0][0])
        predicted = "plant" if score >= 0.5 else "non_plant"
        print(f"{filename}: {predicted} (plant score: {score:.3f})")
