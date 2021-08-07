import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import numpy as np
from PIL import Image, ImageOps

def process(img):
    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = img
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    # Load the image into the array
    data[0] = normalized_image_array
    return data

def get_pred_label(prediction_probabilities, unique_breeds):
    return unique_breeds[np.argmax(prediction_probabilities)]

def make_prediction(img):
    labels_csv = pd.read_csv("app/labels/labels.csv")
    labels = labels_csv["breed"].to_numpy()
    breeds = np.unique(labels)
    model = tf.keras.models.load_model("app/model/dog_breed.h5",
                                     custom_objects={"KerasLayer":hub.KerasLayer}) # Telling we have a custom piece too

    data = process(img)
    preds = model.predict(data)
    custom_pred_labels = [get_pred_label(preds, breeds)]

    return custom_pred_labels