import os
from pathlib import Path

import numpy as np
import tensorflow as tf


def make_tensor_bw(tensor, treshold=0.25):
    # Convert the RGB values to grayscale (brightness) using the luminosity method
    brightness = np.dot(tensor[...,:3], [0.2989, 0.5870, 0.1140])
    mask = brightness >= treshold

    rgb_out = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    rgb_out[mask] = [255, 255, 255]
    return rgb_out

if __name__ == "__main__":
    catalog_path = Path("coin_catalog/augmented")
    shape = (128, 128)

    testrun_name = "small"


    """ Directory for storing files """
    if not os.path.exists(output_dir := Path(os.path.dirname(__file__), "predict")):
        os.makedirs(output_dir)

    model_path = Path(os.path.dirname(__file__), f"trained/{testrun_name}/keras_{testrun_name} - checkpoint2.keras")
    model = tf.keras.models.load_model(model_path)

    # input_path = "coin_catalog/France/2 Franc/1917/uncropped/1.png"
    input_path = "coin_catalog/augmented\images\(Czech Republic, 1 Koruna, 2018)/3_75.png"
    x = tf.io.read_file(input_path)
    x = tf.image.decode_image(x, channels=3)
    x.set_shape([None, None, 3])
    x = tf.image.resize(x, shape)
    x = x / 255
    x = tf.expand_dims(x, axis=0) # Batch enpacking

    y = model.predict(x)
    y = y[0]  # Remove batch dimension
    y = y[:, :, :3]
    y = np.clip(y * 255, 0, 255).astype(np.uint8)
    y = make_tensor_bw(y, treshold=0.1)


    # Save the image using TensorFlow's image encoding
    y_tensor = tf.convert_to_tensor(y, dtype=tf.uint8)
    encoded_png = tf.image.encode_png(y_tensor)  # Encode the image
    output_path = Path(output_dir, "predict", "output_image.png")
    tf.io.write_file(str(output_path), encoded_png)
    print(f"Image saved to {output_path}")
