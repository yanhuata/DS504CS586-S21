# Individual Project 4
# Meta-Learning and Few Shot Learning
#### Due Date
* Thursday, Apr 23, 2020 (23:59)

#### Total Points
* 100 (One Hundred)

## Goal
In project 2, you were given a bunch of drivers and their trajectories to build a model to classify which driver a given trajectory belongs to. In this project, we will give you a harder task. In project 2, the training data contain 5 drivers and 6-month trajectories for each driver. In this task, however, the trainning data contain 500 drivers and only 5-day trajectories for each driver. In this task, you should use meta-learning and/or few shot learning to build the classification model. The model for each driver can be a binary classification, which takes two trajectories as input and predict whether these two trajectories belongs to the same driver. 

## Current Leaderboard
| rank | Name | Accuracy |
|---|---|---|
|   |   | |
|   |    |    |
|  |      |   |

## Evaluation
To evaluation your submission, a seperate test dataset will be held. For each driver, the test data will contains 10 different trajectories. We will randomly generate 20,000 trajectory pairs and use them to evaluate your submitted model. Like project 2, you should submit a evaluation.py file containing how to process the model and how to run prediction. 

## Deliverables & Grading
* PDF Report (50%) [template](https://www.acm.org/binaries/content/assets/publications/taps/acm_submission_template.docx)
    * proposal
    * methodology
    * empirical results and evaluation
    * conslusion
    
* Python Code (50%)
    * Code is required to avoid plagiarism.
    * The submission should contain a python file named "evaluation.py" to help evaluate your model. 
    * The evluation.py should follow the format in the Submission Guideline section. 
    * Evaluation criteria.
      | Percentage | Accuracy |
      |---|---|
      | 100 | 0.6 |
      | 90 | 0.55 |
      | 80 | 0.5 |
      | 70 | 0.45|
      | 60 | 0.4 |
* Grading:
  * Total (100):
    * Code (50) + Report (50)

  * Code (50):
    * accuracy >= 0.60: 50
    * accuracy >= 0.55: 45
    * accuracy >= 0.50: 40
    * accuracy >= 0.45: 35
    * accuracy >= 0.40: 30

  * Report (50):
    1. Introduction & Proposal (5)
    2. Methodology (20):
        a. Data processing (5)
        b. Feature generation (5)
        c. Network structure (5)
        d. Training & validation process (5)
    3. Evaluation & Results (20):
        a. Training & validation results (10)
        b. Performance comparing to your baselines (maybe different network structure) (5)
        c. Hyperparameter (learning rate, dropout, activation) (5)
    4. Conclusion (5)
   * Bonus (5):
   
     5 bonus points for the top 3 on the leader board.

## Project Guidelines

#### Dataset Description
The data is binary pickled file. The data is stored in a dictionary, in which the key is ID of a driver and value is list of his/her trajectories. For each trajectory, the basic element is similar to project 2. Each element in the trajectory is in the following format, [ plate, longitude, latitude, second_since_midnight, status, time ]. Data can be found at [Google Drive](https://drive.google.com/file/d/1aHGJx2KtzjCRlfPYefPaGPl3e5lyJTX-/view?usp=sharing). The training data contain **500** drivers and **5**-day trajectories for each driver.
#### Feature Description 
* **Plate**: Plate means the taxi's plate. In this project, we change them to 0~500 to keep anonymity. Same plate means same driver, so this is the target label for the classification. 
* **Longitude**: The longitude of the taxi.
* **Latitude**: The latitude of the taxi.
* **Second_since_midnight**: How many seconds has past since midnight.
* **Status**: 1 means taxi is occupied and 0 means a vacant taxi.
* **Time**: Timestamp of the record.

#### Problem Definition
Given a full-day trajectory of a taxi and a driver id, you need to predict whether the the given trajectory belongs to that driver. 

#### Evaluation 
Two days of trajectories will be used to evaluate your submission. And test trajectories are not in the data/ folder. However, we have provided a validation dataset. The validate_set.pkl contains validation data and validate_label.pkl contains labels. Same as usual, you can use pickle.load() function to load the dataset and evaluate your model. 
##### Feature Description of validation data
* **Longitude**: The longitude of the taxi.
* **Latitude**: The latitude of the taxi.
* **Second_since_midnight**: How many seconds has past since midnight.
* **Status**: 1 means taxi is occupied and 0 means a vacant taxi.
* **Time**: Timestamp of the record.

#### Submission Guideline
To help better and fast evaluate your model, please submit a separate python file named "evaluation.py". This file should contain two functions.
* **Data Processing**
  ```python
  def processs_data(traj_1, traj_2):
    """
    Input:
        Traj: a list of list, contains one trajectory for one driver 
        example:[[114.10437, 22.573433, '2016-07-02 00:08:45', 1],
           [114.179665, 22.558701, '2016-07-02 00:08:52', 1]]
    Output:
        Data: any format that can be consumed by your model.
    
    """
    return data
  ```
* **Model Prediction**
    ```python
    def run(data, model):
        """
        
        Input:
            Data: the output of process_data function.
            Model: your model.
        Output:
            prediction: the predicted label(plate) of the data, an int value.
        
        """
        return prediction
  ```

## Some Tips
Setup information could also be found in the [slides](https://docs.google.com/presentation/d/1nFZtev4PxJjbxPxEv06YIQwoIUjd7FEcAtrR5JIutGE/edit?usp=sharing)
* Anaconda and virtual environment set tup
   * [Download and install anaconda](https://www.anaconda.com/distribution/)
   * [Create a virtual environment with commands](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands)
* Deep learning package
   * [Keras](https://keras.io/). Keras is a high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano. It is easy to learn and use. If you are new to deep learning, you can try to use Keras to get started. [Naive Tutorial](https://github.com/yanhuata/DS504CS586-S20/blob/master/project2/keras_tutorial.ipynb)
   * [Pytorch](https://pytorch.org/tutorials/)
   * [Tensorflow](https://www.tensorflow.org/tutorials)
   * [MxNet](https://mxnet.apache.org/)
* Open source GPU
   * [Using GPU on Google Cloud](https://github.com/yanhuata/DS504CS586-S20/blob/master/project2/keras_tutorial.ipynb)
   * [Cuda set up for Linux](https://docs.google.com/document/d/1rioVwqvZCbn58a_5wqs5aT3YbRsiPXs9KmIuYhmM1gY/edit?usp=sharing)
   * [Google colab](https://colab.research.google.com/notebooks/gpu.ipynb)
   * [Kaggle](https://www.kaggle.com/dansbecker/running-kaggle-kernels-with-a-gpu)
* **Keywords**. 
   * If you are wondering where to start, you can try to search "sequence classification", "sequence to sequence" or "sequence embedding" in Google or Github, this might provide you some insights.
