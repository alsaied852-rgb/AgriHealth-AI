import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from pathlib import Path
import numpy as np

MODEL_PATH = "leaf_detector_augmented.keras"
TEST_DIR = "data/real_binary/test"
IMG_SIZE = (160, 160)

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

correct = 0
total = 0

for class_name in ["non_plant", "plant"]:
    folder = Path(TEST_DIR) / class_name
    expected_plant = class_name == "plant"
    class_correct = 0
    class_total = 0

    for path in folder.rglob("*"):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue

        image = tf.keras.utils.load_img(
            path, target_size=IMG_SIZE
        )
        array = tf.keras.utils.img_to_array(image)
        array = preprocess_input(array)
        array = np.expand_dims(array, axis=0)

        score = float(model.predict(array, verbose=0)[0][0])
        predicted_plant = score >= 0.5
        is_correct = predicted_plant == expected_plant

        total += 1
        class_total += 1
        correct += int(is_correct)
        class_correct += int(is_correct)

        print(
            f"{class_name}: {'CORRECT' if is_correct else 'WRONG'} "
            f"| plant score={score:.1%} | {path.name}"
        )

    print(f"{class_name} accuracy: {class_correct}/{class_total}")

print(f"\nOverall test accuracy: {correct}/{total} ({correct/total:.1%})")
