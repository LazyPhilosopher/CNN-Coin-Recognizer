import json
from pathlib import Path
import tensorflow as tf

import tensorflow.python.keras as keras


def resize_image(image, target_size=(128, 128)):
    return tf.image.resize(image, target_size, method=tf.image.ResizeMethod.BILINEAR)


if __name__ == "__main__":
    testrun_name = "micro"
    test_dir = Path("coin_catalog/augmented_micro/Czech Republic/50 Korun/2008")
    model_path = Path(f"trained", "models", f"keras_{testrun_name}.keras")
    shape = (128, 128)

    with open(f"trained/enumerations/train_enum_{testrun_name}.json") as f:
        enumerations = json.load(f)



    model = tf.keras.models.load_model(model_path)

    for filepath in test_dir.iterdir():
        filename = filepath.parts[-1]
        image = tf.io.read_file(str(filepath))
        image = tf.image.decode_png(image, channels=3)  # Decode as RGB
        image = tf.image.convert_image_dtype(image, tf.float32)  # Normalize to [0, 1]
        image = resize_image(image, target_size=shape)

        image = tf.expand_dims(image, axis=0)

        predictions = model.predict(image)

    # Output predictions
        predicted_class = tf.argmax(predictions, axis=1).numpy()[0]
        confidence = tf.reduce_max(predictions).numpy()

        print(f"Predicted Class: {enumerations[str(predicted_class)]}")
        print(f"Confidence: {confidence:.2f}")