import os
import numpy as np
import tensorflow as tf
from PIL import Image

MODEL_PATH = "leaf_detector_real.keras"
TEST_DIR = "data/real_binary/test"
IMG_SIZE = (160, 160)

model = tf.keras.models.load_model(MODEL_PATH)

correct = 0
total = 0

for label in ["non_plant", "plant"]:
    folder = os.path.join(TEST_DIR, label)
    files = [
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    for filename in files:
        path = os.path.join(folder, filename)
        try:
            image = Image.open(path).convert("RGB").resize(IMG_SIZE)
            image = np.array(image, dtype=np.float32)
            image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
            score = float(model.predict(np.expand_dims(image, 0), verbose=0)[0][0])

            predicted = "plant" if score >= 0.5 else "non_plant"
            actual = label
            is_correct = predicted == actual
            correct += int(is_correct)
            total += 1

            status = "OK" if is_correct else "WRONG"
            print(
                f"{status} | actual={actual:9s} | "
                f"predicted={predicted:9s} | score={score:.3f} | {filename}"
            )
        except Exception as e:
            print(f"Could not test {filename}: {e}")

print(f"\nCorrect: {correct}/{total}")
if total:
    print(f"Test accuracy: {100 * correct / total:.1f}%")
