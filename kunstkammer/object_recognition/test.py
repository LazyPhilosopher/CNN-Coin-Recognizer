import os
from pathlib import Path

import numpy as np
import tensorflow as tf

from core.utilities.helper import get_directories

if __name__ == "__main__":
    catalog_path = Path("coin_catalog/augmented")
    shape = (128, 128)

    testrun_name = "test"

    # enum_path = f"trained/enumerations/train_enum_{testrun_name}.json"
    enumerations = [str(coin.stem) for coin in get_directories(Path(catalog_path, "images"))]


    model_path = Path(os.path.dirname(__file__), f"trained/object_recognition_{testrun_name}.keras")
    model = tf.keras.models.load_model(model_path)

    # input_path = "coin_catalog/France/2 Franc/1917/uncropped/1.png"
    # input_path = "coin_catalog/augmented\images\(Czech Republic, 2 Koruny, 2022)/9_6.png"
    input_path = "coin_catalog/augmented/images/(Czech Republic, 50 Korun, 2008)/0_10.png"

    file_path = os.path.join(input_path)
    input_image = tf.keras.utils.load_img(file_path, target_size=shape)
    input_image_array = tf.keras.utils.img_to_array(input_image)
    input_image_exp_dim = tf.expand_dims(input_image_array, 0)

    predictions = model.predict(input_image_exp_dim)
    result = tf.nn.softmax(predictions)
    print(f"{file_path} -> {enumerations[np.argmax(result)]}")

