import keras
import matplotlib.pyplot as plt
from keras.models import Sequential,Model,model_from_json
import numpy as np


def load_model():
    '''
    This function is used to load model, codes below are based on template.py.
    Please modify this function based on your own codes.
    '''
    with open("generator.json", "r") as json_file:
        md_json = json_file.read()
    t = model_from_json(md_json)
    t.load_weights("generator.h5")
    return t

def generate_image(model):
    '''
    Take the model as input and generate one image, codes below are based on template.py.
    Please modify this function based on your own codes.
    '''
    # Set the dimensions of the noise
    z_dim = 100
    z = np.random.normal(size=[1, z_dim])
    generated_images = g.predict(z)
    return generated_images

if __name__ == "__main__":
    model = load_model()
    image = generate_image(model)