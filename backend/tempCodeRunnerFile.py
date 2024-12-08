from django.shortcuts import render
from django.http import JsonResponse
from tensorflow.keras.models import load_model
from django.conf import settings
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import pandas as pd
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile

import sys

sys.path.append("../food_recognition")
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'food_recognition.settings')
# Load model and nutrition data
model_path = os.path.join(settings.BASE_DIR, 'backend/models', 'FV.h5')
model = load_model(model_path)
nutrition_data = pd.read_csv(os.path.join(settings.BASE_DIR, 'nutrition101.csv'))

LABELS = ['apple pie',
         'baby back ribs',
         'baklava',
         'beef carpaccio',
         'beef tartare',
         'beet salad',
         'beignets',
         'bibimbap',
         'bread pudding',
         'breakfast burrito',
         'bruschetta',
         'caesar salad',
         'cannoli',
         'caprese salad',
         'carrot cake',
         'ceviche',
         'cheese plate',
         'cheesecake',
         'chicken curry',
         'chicken quesadilla',
         'chicken wings',
         'chocolate cake',
         'chocolate mousse',
         'churros',
         'clam chowder',
         'club sandwich',
         'crab cakes',
         'creme brulee',
         'croque madame',
         'cup cakes',
         'deviled eggs',
         'donuts',
         'dumplings',
         'edamame',
         'eggs benedict',
         'escargots',
         'falafel',
         'filet mignon',
         'fish and_chips',
         'foie gras',
         'french fries',
         'french onion soup',
         'french toast',
         'fried calamari',
         'fried rice',
         'frozen yogurt',
         'garlic bread',
         'gnocchi',
         'greek salad',
         'grilled cheese sandwich',
         'grilled salmon',
         'guacamole',
         'gyoza',
         'hamburger',
         'hot and sour soup',
         'hot dog',
         'huevos rancheros',
         'hummus',
         'ice cream',
         'lasagna',
         'lobster bisque',
         'lobster roll sandwich',
         'macaroni and cheese',
         'macarons',
         'miso soup',
         'mussels',
         'nachos',
         'omelette',
         'onion rings',
         'oysters',
         'pad thai',
         'paella',
         'pancakes',
         'panna cotta',
         'peking duck',
         'pho',
         'pizza',
         'pork chop',
         'poutine',
         'prime rib',
         'pulled pork sandwich',
         'ramen',
         'ravioli',
         'red velvet cake',
         'risotto',
         'samosa',
         'sashimi',
         'scallops',
         'seaweed salad',
         'shrimp and grits',
         'spaghetti bolognese',
         'spaghetti carbonara',
         'spring rolls',
         'steak',
         'strawberry shortcake',
         'sushi',
         'tacos',
         'octopus balls',
         'tiramisu',
         'tuna tartare',
         'waffles']


import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
# image_path = 'momo.jpg'
# img = load_img(image_path, target_size=(224, 224))
# img_array = img_to_array(img)
# img_array = np.expand_dims(img_array, axis=0)
# img_array = img_array / 255.0


# predictions = model.predict(img_array)


# predicted_index = np.argmax(predictions)
# predicted_label = LABELS[predicted_index]
# print(predicted_label)


# food_info = nutrition_data[nutrition_data['name'] == predicted_label]
# calories = food_info['calories']
# protein = food_info['protein']
# calcium = food_info['calcium']
# carbs = food_info['carbohydrates']
# fat = food_info['fat']

# print(f"Estimated Calories: {calories}")
# print(f"Estimated Protein: {protein}")
# print(f"Estimated calcium: {calcium}")
# print(f"Estimated carbohydrates: {carbs}")
# print(f"Estimated fat: {fat}")


def predict_food(request):
    if request.method == 'POST' and request.FILES['image']:
        image = request.FILES['image']

        temp_file_path = default_storage.save(f'temp/{image.name}', ContentFile(image.read()))

        temp_file_path_full = os.path.join(settings.MEDIA_ROOT, temp_file_path)
        img = load_img(temp_file_path_full, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions)
        predicted_label = LABELS[predicted_index]
        print(predicted_label)

        food_info = nutrition_data[nutrition_data['name'] == predicted_label]
        calories = food_info['calories']
        protein = food_info['protein']
        calcium = food_info['calcium']
        carbs = food_info['carbohydrates']
        fat = food_info['fat']

        print(f"Estimated Calories: {calories}")
        print(f"Estimated Protein: {protein}")
        print(f"Estimated calcium: {calcium}")
        print(f"Estimated carbohydrates: {carbs}")
        print(f"Estimated fat: {fat}")

        default_storage.delete(temp_file_path)

       