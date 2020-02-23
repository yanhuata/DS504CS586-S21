#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 18:30:44 2019

@author: Huimin Ren
"""

from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Activation
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt

# Load data
(X_train, _), (_, _) = mnist.load_data()

# Preprocessing
X_train = X_train.reshape(-1, 784)
X_train = X_train.astype('float32')/255

# Set the dimensions of the noise
z_dim = 100

# Optimizer
adam = Adam(lr=0.0002, beta_1=0.5)

# Generator
g = Sequential()
g.add(Dense(256,input_dim = z_dim))
g.add(Activation("relu"))
g.add(Dense(784, activation='sigmoid')) 
#g.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

# Discrinimator
d = Sequential()
d.add(Dense(256))
d.add(Activation("relu"))
d.add(Dense(1, activation='sigmoid'))
d.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

# GAN
d.trainable = False
inputs = Input(shape=(z_dim, ))
hidden = g(inputs)
output = d(hidden)
gan = Model(inputs, output)
gan.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])


# Training
def train(epochs=1, plt_frq=1, BATCH_SIZE=128):
    batchCount = int(X_train.shape[0] / BATCH_SIZE)
    print('Epochs:', epochs)
    print('Batch size:', BATCH_SIZE)
    print('Batches per epoch:', batchCount)
    
    for e in (range(1, epochs+1)):
        print("Epoch:",e)
        for _ in range(batchCount):  
            # Create a batch by drawing random index numbers from the training set
            image_batch = X_train[np.random.randint(0, X_train.shape[0], size=BATCH_SIZE)]
            # Create noise vectors for the generator
            noise = np.random.normal(0, 1, size=(BATCH_SIZE, z_dim))
            
            # Generate the images from the noise
            generated_images = g.predict(noise)
            X = np.concatenate((image_batch, generated_images))
            # Create labels
            y = np.zeros(2*BATCH_SIZE)
            y[:BATCH_SIZE] = 1
    
            # Train discriminator on generated images
            d.trainable = True
            d_loss = d.train_on_batch(X, y)
    
            # Train generator
            noise = np.random.normal(0, 1, size=(BATCH_SIZE, z_dim))
            y2 = np.ones(BATCH_SIZE)
            d.trainable = False
            g_loss = gan.train_on_batch(noise, y2)

# serialize model to JSON
model_json = g.to_json()
with open("generator.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
g.save_weights("generator.h5")
print("Saved model to disk")

# Generate images
np.random.seed(504)
h = w = 28
num_gen = 25

z = np.random.normal(size=[num_gen, z_dim])
generated_images = g.predict(z)

# plot of generation
n = np.sqrt(num_gen).astype(np.int32)
I_generated = np.empty((h*n, w*n))
for i in range(n):
    for j in range(n):
        I_generated[i*h:(i+1)*h, j*w:(j+1)*w] = generated_images[i*n+j, :].reshape(28, 28)

plt.figure(figsize=(4, 4))
plt.axis("off")
plt.imshow(I_generated, cmap='gray')
plt.show()
 