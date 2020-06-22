'''
Predict the top_k possible classes by a given neural network model.

Args:
    image_path: string; path of the image file
    model: tensorflow neural network model
    top_k (optional): float; the top k possible classes; default = 3
    category_names (optional): json file; path to a JSON file mapping labels to flower names

Prints: 
    probs: numpy array; probability of top_k classes
    classes: list; index or name of the top_k classes
'''
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import argparse
import json
from image_preprocessing import process_image
from PIL import Image


parser = argparse.ArgumentParser(description='Predict the (top k) most likely classes of the input image.')

parser.add_argument('image_path', help= 'provide the path of the image file')
parser.add_argument('saved_model', help = 'provide the path of the neural network model')
parser.add_argument('--top_k', default = 1, type = int, help = 'the top k most likely classes')
parser.add_argument('--category_names', default = None, help  = 'provide the path of the json file stroing the category names')

args = parser.parse_args()

image = Image.open(args.image_path)
image = np.asarray(image)
image = process_image(image)
image = np.expand_dims(image, axis = 0)

model = tf.keras.models.load_model(args.saved_model, custom_objects={'KerasLayer':hub.KerasLayer})
ps = model.predict(image)
classes = np.argsort(ps[0])[-args.top_k:] 
probs = ps[0][classes]
classes = [str(x+1) for x in classes]
if args.category_names:
    with open(args.category_names, 'r') as f:
        class_names = json.load(f)
        classes  = [class_names[x] for x in classes]
        
print('\nThe top {} most likely classes are {}, with the probabilities: {}'.format(args.top_k, classes, probs))


