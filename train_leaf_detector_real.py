import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

DATA_DIR = "data/real_binary/train"
IMG_SIZE = (160, 160)
BATCH_SIZE = 8
SEED = 42

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
)

print("Class order:", train_ds.class_names)
print("Expected: non_plant = 0, plant = 1")

augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.10),
    layers.RandomContrast(0.10),
])

train_ds = train_ds.map(
    lambda x, y: (preprocess_input(x), y)
).prefetch(tf.data.AUTOTUNE)

val_ds = val_ds.map(
    lambda x, y: (preprocess_input(x), y)
).prefetch(tf.data.AUTOTUNE)

base = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(160, 160, 3),
)
base.trainable = False

inputs = layers.Input(shape=(160, 160, 3))
x = augmentation(inputs)
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

model = models.Model(inputs, outputs)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
)

model.save("leaf_detector_augmented.keras")
print("Saved candidate model: leaf_detector_augmented.keras")