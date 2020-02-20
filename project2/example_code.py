# coding: utf-8
import pandas as pd
import numpy as np
import pickle
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam

data = pd.read_csv('data/data.csv')


#generate trajectories and label
def aggr(data):
    traj_raw = data.values[:,1:]
    traj = np.array(sorted(traj_raw,key = lambda d:d[2]))
    label = data.iloc[0][0]
    return [traj,label]
processed_data = data.groupby('plate').apply(aggr)

#generate some features 
#you don't have to generate features as soon as you can have an accurate prediction
training = []
labels = []
for traj in processed_data:
    feature = [len(traj[0]),sum(traj[0][:,-1])]
    label = traj[1]
    training.append(feature)
    labels.append(label)

#define your model
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=2))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(np.array(training), np.array(labels), epochs=10, batch_size=32)

#dump model to a pickle file
pickle.dump(model,open('dummy_model.pkl','wb'))