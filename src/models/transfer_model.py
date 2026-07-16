"""Transfer learning model: pretrained VGG16 (ImageNet) + custom classification head.

Spectrograms are single-channel; VGG16 expects 3-channel 224x224 input, so
the input pipeline replicates the channel and resizes before the frozen base.
"""

import keras
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16

TRANSFER_INPUT_SIZE = (224, 224)

_VGG16_BGR_MEAN = (103.939, 116.779, 123.68)


@keras.saving.register_keras_serializable(package="music_genre")
class VGG16Preprocess(layers.Layer):
    """Replicates keras.applications.vgg16.preprocess_input (mode='caffe')
    as plain layer ops, so the model serializes cleanly (a Lambda wrapping
    a Python function is not safely deserializable in Keras 3)."""

    def call(self, x):
        x = x * 255.0
        x = x[..., ::-1]  # RGB -> BGR
        return x - tf.constant(_VGG16_BGR_MEAN, dtype=x.dtype)


def build_transfer_model(input_shape=(128, 128, 1), num_classes=10, fine_tune_at=None):
    """Build a VGG16-based transfer learning model.

    `fine_tune_at`: if None, the VGG16 base is fully frozen (train head only).
    Otherwise, layers from this index onward are unfrozen for fine-tuning.
    """
    base = VGG16(weights="imagenet", include_top=False, input_shape=(*TRANSFER_INPUT_SIZE, 3))

    if fine_tune_at is None:
        base.trainable = False
    else:
        base.trainable = True
        for layer in base.layers[:fine_tune_at]:
            layer.trainable = False

    inputs = layers.Input(shape=input_shape)
    x = layers.Resizing(*TRANSFER_INPUT_SIZE)(inputs)
    x = layers.Concatenate(axis=-1)([x, x, x])  # 1 -> 3 channels
    x = VGG16Preprocess()(x)  # [0,1] -> VGG16's expected preprocessing
    x = base(x, training=False if fine_tune_at is None else None)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="vgg16_transfer")

    lr = 1e-3 if fine_tune_at is None else 1e-5
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":
    build_transfer_model().summary()
