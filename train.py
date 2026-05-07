import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
from sklearn.metrics import classification_report, confusion_matrix

from model import build_cnn_rnn_model, CLASS_NAMES, IMAGE_SIZE

EPOCHS     = 50
BATCH_SIZE = 64
SAVE_DIR   = "saved_model"
os.makedirs(SAVE_DIR, exist_ok=True)

print("Loading CIFAR-10 dataset …")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32")  / 255.0

y_train_oh = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test_oh  = tf.keras.utils.to_categorical(y_test,  num_classes=10)

print(f"  Train: {x_train.shape}   Test: {x_test.shape}")

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
], name="augmentation")

print("Building CNN+RNN model …")
model = build_cnn_rnn_model()
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    EarlyStopping(
        monitor="val_accuracy", patience=8,
        restore_best_weights=True, verbose=1
    ),
    ModelCheckpoint(
        filepath=os.path.join(SAVE_DIR, "best_model.keras"),
        monitor="val_accuracy", save_best_only=True, verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=4,
        min_lr=1e-6, verbose=1
    ),
]

print("\nStarting training …")

train_ds = (
    tf.data.Dataset.from_tensor_slices((x_train, y_train_oh))
    .shuffle(10_000)
    .batch(BATCH_SIZE)
    .map(lambda x, y: (data_augmentation(x, training=True), y),
         num_parallel_calls=tf.data.AUTOTUNE)
    .prefetch(tf.data.AUTOTUNE)
)

val_ds = (
    tf.data.Dataset.from_tensor_slices((x_test, y_test_oh))
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
)

print("\nEvaluating on test set …")
loss, acc = model.evaluate(val_ds, verbose=0)
print(f"  Test accuracy : {acc*100:.2f}%")
print(f"  Test loss     : {loss:.4f}")

y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
print("\nClassification report:")
print(classification_report(y_test.flatten(), y_pred, target_names=CLASS_NAMES))

final_path = os.path.join(SAVE_DIR, "cnn_rnn_cifar10.keras")
model.save(final_path)
print(f"\nModel saved → {final_path}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history.history["accuracy"],     label="train acc")
axes[0].plot(history.history["val_accuracy"], label="val acc")
axes[0].set_title("Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].legend()

axes[1].plot(history.history["loss"],     label="train loss")
axes[1].plot(history.history["val_loss"], label="val loss")
axes[1].set_title("Loss")
axes[1].set_xlabel("Epoch")
axes[1].legend()

plt.tight_layout()
curve_path = os.path.join(SAVE_DIR, "training_curves.png")
plt.savefig(curve_path)
print(f"Training curves saved → {curve_path}")
print("\nDone! Now run:  python app.py")