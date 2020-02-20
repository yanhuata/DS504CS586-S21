import pandas as pd
import numpy as np
import pickle
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam

#def run and predict
def process_data(traj):
    feature = [len(traj),sum(traj[:,-1])]
    return feature
def run(data, model):
    return model.predict(np.array([data]))