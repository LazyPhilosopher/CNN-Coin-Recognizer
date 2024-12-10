from pathlib import Path

import numpy as np
import tensorflow as tf
from core.utilities.helper import get_files, get_directories


if __name__=="__main__":
    catalog_path = Path("D:/Projects/bachelor_thesis/OpenCV2-Coin-Recognizer/coin_catalog")
    model_path = Path("")
    model = tf.keras.models.load_model(model_path)

    country_dirs = get_directories(catalog_path)
    country_dirs = [directory for directory in country_dirs if "augmented" not in directory.stem]

    enumerations = [str(coin.stem) for coin in get_directories(Path(catalog_path, "images"))]

    image_dict = {}
    for country_dir in country_dirs:
        for coin_dir in get_directories(country_dir):
            for year_dir in get_directories(coin_dir):
                country = country_dir.stem
                coin_name = coin_dir.stem
                year = year_dir.stem

                image_dict[(country,coin_name,year)] = get_files(Path(year_dir, "uncropped"))
                image_dict_inv = {value: key for key, value in image_dict.items()}

    results = []
    pass_fail = {"pass": 0, "fail": 0}
    for image_path, coin in image_dict_inv.items():

        input_image = tf.keras.utils.load_img(image_path, target_size=(128,128))
        input_image_array = tf.keras.utils.img_to_array(input_image)
        input_image_exp_dim = tf.expand_dims(input_image_array, 0)

        predictions = model.predict(input_image_exp_dim)
        result = tf.nn.softmax(predictions)
        prediction = enumerations[np.argmax(result)]

        if coin == prediction:
            pass_fail["pass"] += 1
            results.append(f"['PASS' ] {coin} -> {prediction}")
        else:
            pass_fail["fails"] += 1
            results.append(f"['FAIL' ] {coin} -> {prediction} [{image_path}]")

    [print(result) for result in results]




    pass
