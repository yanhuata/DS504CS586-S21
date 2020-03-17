# Individual Project 2
# Classification with Deep Learning
#### Due Date
* Thursday, Mar 5th, 2020 (23:59)

#### Total Points
* 100 (One Hundred)

## Goal
In this project, you will be asked to finish a sequence classification task using deep learning. A trajectory data set with five taxi drivers' daily driving trajectories in 6 months will be provided, and the task is to build a model to predict which driver a trajectory belongs to. A trajectory to be classified includes all GPS records of a driver in a day. During the test, we will use data from 5 drivers in 5 days, i.e. there will be 25 labels to be evaluated. You can do anything to preprocess the data before input the data to the neural network, such as extracting features, getting sub-trajectory based on the status, and so on. This project should be completed in Python 3. Keras, Pytorch, and Tensorflow are recommended, but you can make your decision to use other tools like MxNet.

## Current Leaderboard
| rank | Name | Accuracy |
|---|---|---|
|**1**    |**Zhao, Zixuan**    |**0.96**   |
|**1**    |**Catherman, Davis S**    |**0.96**    |
|**3**   |**Robbertz, Andrew**       |**0.92**       |
|4  |Valente, Richard       |0.88       |
|5    |Guan, Yiwen     |0.84    |
|5    |Lu, Senbao|0.84|
|7     |Ziyao Gao|0.80|
|8     |Neville, Cory    |0.76    |
|9    |Sarwar, Atifa   |0.76    |
|10    |Liang, Yueqing  |0.72
|11    |Hou, Songlin    |0.68    |
|12   |Charbonneau, Jack   |0.64    |
|13   |Manyang Sun   |0.64    |
|14   |Capobianco, Michael   |0.64    |


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

## Project Guidelines

#### Dataset Description
| plate | longitute | latitude | time | status |
|---|---|---|---|---|
|4    |114.10437    |22.573433    |2016-07-02 0:08:45    |1|
|1    |114.179665    |22.558701    |2016-07-02 0:08:52    |1|
|0    |114.120682    |22.543751    |2016-07-02 0:08:51    |0|
|3    |113.93055    |22.545834    |2016-07-02 0:08:55    |0|
|4    |114.102051    |22.571966    |2016-07-02 0:09:01    |1|
|0    |114.12072    |22.543716    |2016-07-02 0:09:01    |0|


Above is an example of what the data look like. In the data/ folder, each .csv file is trajectories for 5 drivers in the same day. Data can be found at [Google Drive](https://drive.google.com/open?id=1xfyxupoE1C5z7w1Bn5oPRcLgtfon6xeT)
#### Feature Description 
* **Plate**: Plate means the taxi's plate. In this project, we change them to 0~5 to keep anonymity. Same plate means same driver, so this is the target label for the classification. 
* **Longitude**: The longitude of the taxi.
* **Latitude**: The latitude of the taxi.
* **Time**: Timestamp of the record.
* **Status**: 1 means taxi is occupied and 0 means a vacant taxi.

#### Problem Definition
Given a full-day trajectory of a taxi, you need to predict which taxi driver it belongs to. 

#### Evaluation 
Five days of trajectories will be used to evaluate your submission. And test trajectories are not in the data/ folder. 

#### Submission Guideline
To help better and fast evaluate your model, please submit a separate python file named "evaluation.py". This file should contain two functions.
* **Data Processing**
  ```python
  def processs_data(traj):
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
    def run(data,model):
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
   

