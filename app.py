import tensorflow as tf
import tensorflow_hub as hub
# import streamlit as st
# import matplotlib
# import matplotlib.pyplot as plt
import os
import pandas
import numpy as np
from PIL import Image, ImageOps

# What I have to do:

# 1. Get create_batches
# 2. Get unbatchify
# 3. download the model.h5 thing
# 4. Load it
# 5. Predict
# 6. Plot
# 7. Streamlit stuff


breednames = np.array(['affenpinscher', 'afghan_hound', 'african_hunting_dog', 'airedale',
                       'american_staffordshire_terrier', 'appenzeller',
                       'australian_terrier', 'basenji', 'basset', 'beagle',
                       'bedlington_terrier', 'bernese_mountain_dog',
                       'black-and-tan_coonhound', 'blenheim_spaniel', 'bloodhound',
                       'bluetick', 'border_collie', 'border_terrier', 'borzoi',
                       'boston_bull', 'bouvier_des_flandres', 'boxer',
                       'brabancon_griffon', 'briard', 'brittany_spaniel', 'bull_mastiff',
                       'cairn', 'cardigan', 'chesapeake_bay_retriever', 'chihuahua',
                       'chow', 'clumber', 'cocker_spaniel', 'collie',
                       'curly-coated_retriever', 'dandie_dinmont', 'dhole', 'dingo',
                       'doberman', 'english_foxhound', 'english_setter',
                       'english_springer', 'entlebucher', 'eskimo_dog',
                       'flat-coated_retriever', 'french_bulldog', 'german_shepherd',
                       'german_short-haired_pointer', 'giant_schnauzer',
                       'golden_retriever', 'gordon_setter', 'great_dane',
                       'great_pyrenees', 'greater_swiss_mountain_dog', 'groenendael',
                       'ibizan_hound', 'irish_setter', 'irish_terrier',
                       'irish_water_spaniel', 'irish_wolfhound', 'italian_greyhound',
                       'japanese_spaniel', 'keeshond', 'kelpie', 'kerry_blue_terrier',
                       'komondor', 'kuvasz', 'labrador_retriever', 'lakeland_terrier',
                       'leonberg', 'lhasa', 'malamute', 'malinois', 'maltese_dog',
                       'mexican_hairless', 'miniature_pinscher', 'miniature_poodle',
                       'miniature_schnauzer', 'newfoundland', 'norfolk_terrier',
                       'norwegian_elkhound', 'norwich_terrier', 'old_english_sheepdog',
                       'otterhound', 'papillon', 'pekinese', 'pembroke', 'pomeranian',
                       'pug', 'redbone', 'rhodesian_ridgeback', 'rottweiler',
                       'saint_bernard', 'saluki', 'samoyed', 'schipperke',
                       'scotch_terrier', 'scottish_deerhound', 'sealyham_terrier',
                       'shetland_sheepdog', 'shih-tzu', 'siberian_husky', 'silky_terrier',
                       'soft-coated_wheaten_terrier', 'staffordshire_bullterrier',
                       'standard_poodle', 'standard_schnauzer', 'sussex_spaniel',
                       'tibetan_mastiff', 'tibetan_terrier', 'toy_poodle', 'toy_terrier',
                       'vizsla', 'walker_hound', 'weimaraner', 'welsh_springer_spaniel',
                       'west_highland_white_terrier', 'whippet',
                       'wire-haired_fox_terrier', 'yorkshire_terrier'])
IMAGE_SIZE = 224


def process_image(img):
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = img
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    data[0] = normalized_image_array
    return data


def preds_to_text(prediction_proba):
    return breednames[np.argmax(prediction_proba)]


def load_model(model_path):
    print(f'Loading model from: {model_path}...')
    model = tf.keras.models.load_model(model_path,
                                       custom_objects={"KerasLayer": hub.KerasLayer})
    return model


def predict_custom(image):
    model = load_model(model_path='dog_breed.h5.h5')
    custom_data = process_image(image)
    custom_preds = model.predict(custom_data)
    conf = f'{np.max(custom_preds[0]) * 100:.2f}%'
    custom_preds_labels = preds_to_text(custom_preds)

    return image, custom_preds_labels, conf