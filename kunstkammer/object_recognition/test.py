import os
from pathlib import Path

import numpy as np
import tensorflow as tf

from core.utilities.helper import get_directories

if __name__ == "__main__":
    catalog_path = Path("coin_catalog/augmented")
    shape = (128, 128)

    testrun_name = "ResNet50"

    # enum_path = f"trained/enumerations/train_enum_{testrun_name}.json"
    enumerations = [str(coin.parts[-1]) for coin in get_directories(Path(catalog_path, "images"))]

    model_path = Path(os.path.dirname(__file__), f"trained/object_recognition_{testrun_name}.keras")
    model = tf.keras.models.load_model(model_path)

    samples = [
        "coin_catalog/augmented/crops/(Czech Republic, 50 Korun, 2008)/0_10.png",
        "coin_catalog/augmented/crops/(Czech Republic, 1 Koruna, 2018)/0_19.png",
        "coin_catalog/augmented/crops/(USA, 2.5 Dollar, 1909)/1_7.png",
        "coin_catalog/augmented/crops/(Great britain, 0.5 Souvereign, 1906)/0_13.png",
        "coin_catalog/augmented/crops/(Iran, 0.5 Souvereign, 1925)/4_3.png",
        "coin_catalog/augmented/crops/(Austria-Hungary, 20 Korona, 1893)/0_8.png",
        "coin_catalog/augmented/crops/(Czech Republic, 10 Korun, 2020)/0_3.png",
        "coin_catalog/augmented/crops/(France, 2 Franc, 1917)/0_20.png",
        "coin_catalog/augmented/crops/(Czech Republic, 5 Korun, 2002)/0_17.png",
        "coin_catalog/augmented/crops/(India, 1 Rupee, 1840)/4_26.png",
        "coin_catalog/augmented/crops/(Austria-Hungary, 20 Korona, 1893)/0_11.png",
    ]

    results = []
    for sample_path in samples:
        input_image = tf.keras.utils.load_img(sample_path, target_size=shape)
        input_image_array = tf.keras.utils.img_to_array(input_image)
        input_image_exp_dim = tf.expand_dims(input_image_array, 0)

        predictions = model.predict(input_image_exp_dim)
        result = tf.nn.softmax(predictions)

        sample_dir = Path(sample_path).parts[-2]
        predict_dir = enumerations[np.argmax(result)]

        results.append(f"[{'PASS' if sample_dir == predict_dir else 'FAIL'}] {sample_dir} -> {predict_dir}")

    [print(result) for result in results]
