import os
from pathlib import Path

import numpy as np

from core.utilities.helper import get_directories, get_files
from kunstkammer.neural_network_playground.classification import ClassificationModel
from kunstkammer.neural_network_playground.core.helper import apply_rgb_mask, load_enumerations
from kunstkammer.neural_network_playground.crop import CropModel
import tensorflow as tf


trained_model_dir = Path(os.path.dirname(__file__), "trained")
catalog_path = Path("D:/Projects/bachelor_thesis/OpenCV2-Coin-Recognizer/coin_catalog/augmented_10")
crop_shape = (128, 128)
classification_shape = (512, 512)

crop_model_name = "crop_model"
classification_model_name = "classification_model"

if __name__ == "__main__":

    crop_model = CropModel(crop_shape)
    classification_model = ClassificationModel(classification_shape)

    # test_samples = [
    #     "(Czech Republic, 50 Korun, 2008)/0_9.png",
    #     "(Czech Republic, 1 Koruna, 2018)/1_8.png",
    #     "(USA, 2.5 Dollar, 1909)/1_7.png",
    #     "(Great britain, 0.5 Souvereign, 1906)/0_3.png",
    #     "(Iran, 0.5 Souvereign, 1925)/4_3.png",
    #     "(Austria-Hungary, 20 Korona, 1893)/0_8.png",
    #     "(Czech Republic, 10 Korun, 2020)/0_3.png",
    #     "(France, 2 Franc, 1917)/0_2.png",
    #     "(Czech Republic, 5 Korun, 2002)/0_7.png",
    #     "(India, 1 Rupee, 1840)/4_6.png",
    #     "(Austria-Hungary, 20 Korona, 1893)/0_1.png",
    # ]

    test_samples = []
    for coin_dir_path in get_directories(Path(catalog_path, "images")):
        # if "augmented" in str(coin_dir_path):
        #     continue
        for image_path in get_files(coin_dir_path):
            test_samples.append(str(image_path.parts[-2] + "/"+ image_path.parts[-1]))

    try:
        crop_model.load_model(Path(trained_model_dir, crop_model_name))
        classification_model.load_model(Path(trained_model_dir, classification_model_name))
        enumerations = load_enumerations(Path(trained_model_dir, classification_model_name))
        print("=== Model Datas loaded ===")

    except Exception as ex:
        print(f"=== Could not load datasets ===")
        print(f"Exception: {ex}")
        exit(-1)


    results = {"pass": [], "fail": []}
    for sample_path in test_samples:
        full_sample_path = str(Path( catalog_path, "images", sample_path))
        small_image = crop_model.load_image(full_sample_path)
        image_full = classification_model.load_image(full_sample_path)

        small_image_batch = tf.expand_dims(small_image, 0)
        bw_mask = crop_model.predict_mask(small_image_batch, threshold=0.003, verbose=0)
        bw_mask = tf.image.resize(bw_mask, classification_shape)

        masked_image = apply_rgb_mask(image_full, bw_mask)
        masked_image_batch = tf.expand_dims(masked_image, 0)

        predictions = classification_model.predict(masked_image_batch)
        result = tf.nn.softmax(predictions)

        true_class = Path(sample_path).parts[0]
        predict_class = enumerations[np.argmax(result)]

        if true_class == predict_class:
            results["pass"].append(sample_path)
        else:
            results["fail"].append(sample_path)

        print(f"[{'PASS' if true_class == predict_class else 'FAIL'}] {true_class} -> {predict_class}")

    print(f"Total Pass/Fail: {len(results['pass'])}/{len(results['fail'])}")






