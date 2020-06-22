'''
Resize the images and normalize the pixel values for future use in trained neural network

Args:
    image: numpy array, pixel data of the image
    image_size: float, the image size required by the neural network
    
Returns: numpy array of resized and normalized image data

'''
import tensorflow as tf
import numpy as np

def process_image(image, image_size = 224):
    image_ts = tf.convert_to_tensor(image)
    image_ts = tf.cast(image, tf.float32)
    image_ts = tf.image.resize(image_ts, (image_size, image_size))
    image_ts /=255
    return image_ts.numpy()