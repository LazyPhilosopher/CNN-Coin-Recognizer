import os
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import Model


shape = (128, 128)

# Load the pre-trained models
crop_model = tf.keras.models.load_model('D:/Projects/bachelor_thesis/OpenCV2-Coin-Recognizer/kunstkammer/background_deletion/trained/keras_test.keras')
classify_model = tf.keras.models.load_model('D:/Projects/bachelor_thesis/OpenCV2-Coin-Recognizer/kunstkammer/object_recognition/trained/object_recognition_ResNet50.keras')

# Freeze the layers if no further training is needed
crop_model.trainable = False
classify_model.trainable = False

# Input for the combined model
input_tensor = tf.keras.Input(shape=(*shape, 3))  # Adjust shape accordingly

# Pass input through the first model (cropping)
cropped_output = crop_model(input_tensor)

# Pass the cropped output through the second model (classification)
classification_output = classify_model(cropped_output)

# Create the composite model
combined_model = Model(inputs=input_tensor, outputs=classification_output, name="Combined_Crop_and_Classify")

# Optionally, compile the combined model (only if you want to train it further)
combined_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Summary of the combined model
combined_model.summary()

# Save the combined model if needed
combined_model.save(str(Path(os.path.dirname(__file__), 'combined_model.keras')))