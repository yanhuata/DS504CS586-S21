# Individual Project 2
# Classification with Deep Learning
#### Due Date
* Thursday Mar 5th, 2020 (23:59)

#### Total Points
* 100 (One Hundred)

## Goal
#

## Deliverables & Grading
* PDF Report (70%) [template](https://www.acm.org/binaries/content/assets/publications/taps/acm_submission_template.docx)
	* proposal
	* methodology
	* empirical results and evaluation
	* conslusion
	
* Python Code (30%)
	* Code is required to avoid plagiarism.
	* The submission should contain a python file named "evaluation.py" to help evluation your model. 
	* The evluation.py should follow the format in the Submission Guideline section. 

## Project Guidelines

#### Dataset Description
| plate | longitute | latitude | time | status |
|---|---|---|---|---|
|4	|114.10437	|22.573433	|2016-07-02 0:08:45	|1|
|1	|114.179665	|22.558701	|2016-07-02 0:08:52	|1|
|0	|114.120682	|22.543751	|2016-07-02 0:08:51	|0|
|3	|113.93055	|22.545834	|2016-07-02 0:08:55	|0|
|4	|114.102051	|22.571966	|2016-07-02 0:09:01	|1|
|0	|114.12072	|22.543716	|2016-07-02 0:09:01	|0|
Above is an example of what the data look like. In the data/ folder, each .csv file is trajectories for 5 drivers in the same day. 
#### Feature Description 
* **Plate**: Plate means the taxi's plate. In this project, we change them to 0~5 to keep anonymity. Same plate means same driver, so this is the target label for the classification. 
* **Longitue**: The longitude of the taxi.
* **Latitude**: The latitude of the taxi.
* **Time**: Timestamp of the record.
* **Status**: 1 means taxi is occupied and 0 means a vacant taxi.

#### Problem Definition
Given a full-day trajectory of a taxi, you need to predict which taxi driver it belongs to. 

#### Evaluation 
5 days of trajectories will be used to evaluate your submission. And test trajectories are not in the data/ folder. 

#### Submission Guideline
To help better and fast evaluate your model, please submit a seperate python file named "evaluation.py". This file should contain two functions.
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
* [Keras](https://keras.io/). Keras is a high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano. It is easy to learn and use. If you are new to deep learning, you can try to use keras to get started. 
* **Keywords**. If you are wondering where to start, you can try to search "sequence classfication", "sequence to sequence" or "sequence embedding" in Google or Github, this might provide you some insights.