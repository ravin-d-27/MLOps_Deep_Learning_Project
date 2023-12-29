# MLOps Project - Deployment of Deep Learning Model

This repository contains the MLOps implementation of Deploying and making a prediction server of a Deep Learning Model using ZenML and MLFlow.

## About the Dataset:
Coronavirus disease (COVID-19) is an infectious disease caused by a newly discovered coronavirus. Most people infected with COVID-19 virus will experience mild to moderate respiratory illness and recover without requiring special treatment. Older people, and those with underlying medical problems like cardiovascular disease, diabetes, chronic respiratory disease, and cancer are more likely to develop serious illness.
During the entire course of the pandemic, one of the main problems that healthcare providers have faced is the shortage of medical resources and a proper plan to efficiently distribute them. In these tough times, being able to predict what kind of resource an individual might require at the time of being tested positive or even before that will be of immense help to the authorities as they would be able to procure and arrange for the resources necessary to save the life of that patient.

The main goal of this project is to build a machine learning model that, given a Covid-19 patient's current symptom, status, and medical history, will predict whether the patient is in high risk or not.

&nbsp; 

Dataset used: https://www.kaggle.com/datasets/meirnizri/covid19-dataset

The model here is used to classify the patients under the following category:

- Values 1 to 3 mean that the patient was diagnosed with covid in different degrees. 
- 4 to 7 means that the patient is not a carrier of covid or that the test is inconclusive.

Here, I have used Artificial Neural Networks model which classifies whether the patient is in the range of 1 to 7. The ANN model is built using TensorFlow Library. Also, the implemented model is a multi-class classification, which means that the model predict more than 1 class or category.


## Requirements:

- Python 3.x
- Linux or Mac Environment (In windows, use WSL. In my case, I have used Linux to develop this)
- Good Specifications to run the Deep Learning Model (GPU is Preferred)

For packages and libraries, refer to requirements.txt file

## Training Pipeline

This script, training_pipeline.py, is a ZenML pipeline that orchestrates the process of training a machine learning model. It includes the following steps:

- Data Ingestion: The run function from steps.ingest_data module is used to ingest data from a given data path.

- Data Cleaning and Splitting: The clean_data function from steps.clean_data module is used to clean the ingested data and transforms that into Features (X) and Target (y). The split_data function from step.clean_data is used to split the data into Training and Testing Set using scikit-learn’s train_test_split function from skeleton.model_selection.

- Model Training: The model_train function from steps.model_train module is used to train the model using the training data.

- Model Evaluation: The model_eval function from steps.model_eval module is used to evaluate the trained model using the testing data.


## Usage

This pipeline is decorated with the @pipeline decorator from ZenML, and takes a single argument: data_path, which is the path to the data to be ingested.

```python
@pipeline
def training_pipeline(data_path: str):
    ...
```

Before running the pipeline, you have to run these commands in order to register this stack to MLFlow


`zenml integration install mlflow -y`
`zenml experiment-tracker register mlflow_tracker --flavor=mlflow`
`zenml model-deployer register mlflow --flavor=mlflow`
`zenml stack register mlflow_stack -a default -o default -d mlflow -e mlflow_tracker --set`
`python run_deployment.py --config deploy` - To Deploy the Model
`python run_deployment.py --config predict`- To Predict the results from the Model

To run this pipeline, you would typically import it in another script and call it with the path to your data:

```python
from training_pipeline import training_pipeline
training_pipeline('path/to/your/data.csv')
```

## Important Commands

- `zenml up` - To turn up the server
- `zenml down` - To turn down the server of zenml
- `zenml disconnect` - to disconnect zenml server
- `zenml init` - To initialize the zenml folder
- `zenml stack describe`- To see the stack description
- `zenml stack list` - Lists down the stack names along with Stack ID and which stack is active
