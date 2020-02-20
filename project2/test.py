import pandas as pd
import numpy as np
import pickle
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
from evaluation import *

model = pickle.load(open('dummy_model.pkl','rb'))
test_data = pickle.load(open('test.pkl','rb'))

for te in test_data:
    dt = process_data(te)
    re = run(dt, model)