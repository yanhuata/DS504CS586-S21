
# Individual Project 3:
# Image Generation with GAN

#### Due Date
* Thursday April 2, 2019 (23:59)

#### Total Points 
* 100 (One Hundred)

## Goal
In this assignment you will be asked to implement a Generative Adversarial Networks (GAN) with [MNIST data set](http://yann.lecun.com/exdb/mnist/). This project will be completed in Python 3 using [Keras](https://keras.io/). 

<img src="https://github.com/yanhuata/DS504CS586-S20/blob/master/project3/pic/goal.png" width="80%">


## Project Guidelines

#### Data set

MNIST is a dataset composed of handwrite numbers and their labels. Each MNIST image is a 28\*28 grey-scale image, which is labeled with an integer value from 0 and 9, corresponding to the actual value in the image. MNIST is provided in Keras as 28\*28 matrices containing numbers ranging from 0 to 255. There are 60000 images and labels in the training data set and 10000 images and labels in the test data set. Since this project is an unsupervised learning project, you can only use the 60000 images for your training. 

#### Installing Software and Dependencies 

* [Install Anaconda](https://docs.anaconda.com/anaconda/install/)
* [Create virtual environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
* Install packages (e.g. pip install keras)

#### Building and Compiling Generator and Discriminator

In Keras, Models are instantiations of the class Sequential. A Keras model [template](template.py), is provided which can be used as starting point for building your model. You can try different layers, such as “Con2D”, different activation functions, such as “tanh”, “leakyRelu”. To compile the model, different optimizer, such as stochastic gradient descent and different loss function are also allowed. The following is the sample code of how to build and compile the models.


```python
# Optimizer
adam = Adam(lr=0.0002, beta_1=0.5)

# Generator
g = Sequential()
g.add(Dense(256,input_dim = z_dim))
g.add(Activation("relu"))
g.add(Dense(784, activation='sigmoid')) 


# Discrinimator
d = Sequential()
d.add(Dense(256,input_dim=784))
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
```

#### Training GAN

You have the option of changing how many epochs to train your model for and how large your batch size is. The following is the sample code of how to train GAN.



```python
# Training
def train(epochs=1, plt_frq=1, BATCH_SIZE=128):
    batchCount = int(X_train.shape[0] / BATCH_SIZE)
    print('Epochs:', epochs)
    print('Batch size:', BATCH_SIZE)
    print('Batches per epoch:', batchCount)
    
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
```

#### Saving Generator

Please use the following code to save the model and weights of your generator.



```python
# serialize model to JSON
model_json = g.to_json()
with open("generator.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
g.save_weights("generator.h5")
```

#### Plotting

Please use the following code to plot the generated images. As for the loss plot of your generator and discriminator during the training, you can plot with your own style. 


```python
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
```

## Deliverables

Please compress all the below files into a zipped file and submit the zip file (firstName_lastName_GAN.zip) to Canvas. 

#### PDF Report
* Set of Experiments Performed: Include a section describing the set of experiments that you performed, what structures you experimented with (i.e., number of layers, number of neurons in each layer), what hyperparameters you varied (e.g., number of epochs of training, batch size and any other parameter values, weight initialization schema, activation function), what kind of loss function you used and what kind of optimizer you used. 
* Special skills: Include the skills which can improve the generation quality. Here are some [tips](https://github.com/soumith/ganhacks) may help.   
* Visualization: Include 25 (5\*5) final generated images which formatted as the example in Goal and a loss plot of the generator and discriminator during your training. For generated images, you need to generated at least one image for each digit. 

#### Python code
* Include model creation, model training, plotting code.

#### Generator Model
* Turn in your best generator saved as “generator.json” and the weights of your generator saved as “generator.h5”.


## Grading

#### Report (70%)

* Set of experiments performed: 30 points
* Special skills: 20 points
* Visualization: 20 points

#### Code (20%) 

You can get full credits if the scripts can run successfully (i.e., TA will test your code with a small data set to see if images can be generated), otherwise you may loss some points based on your error. 

#### Model (10%)

You can get full credits if all the generated images can be recognized, otherwise you may loss some points. Also, the code you submitted should be able to generate all 10 different digits.

## Bonus (10 points)

Generate images from other data source.

* Data set

  Here are some images you may interest. Note other data sets are also allowed.
  
  [Face](https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg)
  
  [Dogs and Cats](https://www.kaggle.com/c/dogs-vs-cats/data)
  
  [Anime](https://drive.google.com/drive/folders/1mCsY5LEsgCnc0Txv0rpAUhKVPWVkbw5I)
  
* Package

  You are allowed to use any deep learning package, such as Tensorflow, Pytorch, etc.
  
* Deliverable

  * Code
  
  * Model
  
  * README file (How to  compile and load the model to generate images)
  
  * 25 generated images

## Tips of Using GPU on Turing Server

* [Apply Turing account](http://arc.wpi.edu/computing/accounts/turing-accounts/)

    * [Turing documentation](http://arc.wpi.edu/cluster-documentation/build/html/batch_manager.html)

* Install anaconda on Turing

    * [Download](https://www.anaconda.com/distribution/#linux) anaconda with Python 3 for Linux installer
    
    * Upload it to your Turing account
    
    * Install anaconda // Linux command: bash \*\*.sh
    
    * Activate anaconda // Linux command: source ~/.bashrc

* [Create virutal environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands )
    * Create virtual environment // Linux command: conda create -n myenv python=3.6
    
    * Activate virtual environment // Linux command: conda activate myenv
    
* Install packages in virtual environment    

A [Tutorial](TuringTutorial.pdf) file is attached which also contains some basic commands for Linux.
