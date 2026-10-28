import os
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("leaf_detector.keras")
base = "data/binary/test"
results = {"plant": {"correct": 0, "total": 0}, "non_plant": {"correct": 0, "total": 0}}

for expected in ["non_plant", "plant"]:
    folder = os.path.join(base, expected)
    for filename in os.listdir(folder):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        path = os.path.join(folder, filename)
        img = tf.keras.utils.load_img(path, target_size=(160, 160))
        arr = tf.keras.utils.img_to_array(img)
        arr = np.expand_dims(arr, axis=0)
        score = float(model.predict(arr, verbose=0)[0][0])
        predicted = "plant" if score >= 0.5 else "non_plant"

        results[expected]["total"] += 1
        if predicted == expected:
            results[expected]["correct"] += 1
        else:
            print(f"Wrong: expected={expected}, predicted={predicted}, file={filename}, score={score:.3f}")

for label, data in results.items():
    print(f"{label}: {data['correct']}/{data['total']} correct")

correct = sum(x["correct"] for x in results.values())
total = sum(x["total"] for x in results.values())
print(f"Overall test accuracy: {correct}/{total} = {correct/total:.1%}")
