import tensorflow as tf
from tensorflow.keras import layers, models

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

NUM_CLASSES = len(CLASS_NAMES)
IMAGE_SIZE = (32, 32)  # CIFAR-10 native size
INPUT_SHAPE = (32, 32, 3)


def build_cnn_rnn_model():
    """
    Hybrid CNN + RNN model for multi-class image classification.
    
    Architecture:
      CNN part  : Extracts spatial features from the image using Conv2D layers.
      Reshape   : Converts 2-D feature maps into a sequence for the RNN.
      RNN part  : LSTM layer learns temporal/sequential patterns in the features.
      Head      : Dense + Softmax for final class probabilities.
    """
    inputs = layers.Input(shape=INPUT_SHAPE, name="image_input")

    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)          # 32x32 → 16x16
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)          # 16x16 → 8x8
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)          # 8x8 → 4x4
    x = layers.Dropout(0.25)(x)

    cnn_out_shape = x.shape[1:]                 # e.g. (4, 4, 128)
    time_steps = cnn_out_shape[0]               # 4
    features   = cnn_out_shape[1] * cnn_out_shape[2]  # 4 * 128 = 512
    x = layers.Reshape((time_steps, features))(x)

    x = layers.LSTM(256, return_sequences=False)(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="CNN_RNN_Classifier")
    return model


if __name__ == "__main__":
    model = build_cnn_rnn_model()
    model.summary()
