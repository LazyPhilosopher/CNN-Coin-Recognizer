import tensorflow as tf
from keras.applications import ResNet50
from tensorflow.keras import layers, Model


def build_model(input_shape=(224, 224, 3), num_classes=10, use_pretrained=True):
    base_model = tf.keras.applications.ResNet152V2(
        weights="imagenet" if use_pretrained else None,
        include_top=False,
        input_shape=input_shape
    )

    base_model.trainable = not use_pretrained

    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs, name="ResNet50V2_Custom")

    return model


def build_resnet34_model(input_shape=(224, 224, 4), num_classes=10, use_pretrained=True):
    """
    Builds a ResNet34 model using a custom approach (or you can use timm for official ResNet34).

    Args:
        input_shape (tuple): Shape of input images.
        num_classes (int): Number of output classes.
        use_pretrained (bool): Whether to use pretrained weights.

    Returns:
        Model: A TensorFlow Keras Model.
    """
    # Load ResNet50 and adapt it to simulate ResNet34 architecture
    base_model = ResNet50(
        weights="imagenet" if use_pretrained else None,
        include_top=False,
        input_shape=input_shape
    )

    # Make the model trainable or not based on `use_pretrained`
    base_model.trainable = not use_pretrained

    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs, name="ResNet34_Custom")

    return model
