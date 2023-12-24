import json
import numpy as np
import pandas as pd

# from materializer.custom_materializer import cs_materializer
from zenml import pipeline, step
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.config import DockerSettings
from zenml.integrations.constants import MLFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)

from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.steps import BaseParameters, Output

from steps.ingest_data import ingest_data
from steps.clean_data import clean_data, split_data
from steps.model_train import model_train
from steps.model_eval import model_eval


from .utils import get_data_for_test


docker_settings = DockerSettings(required_integrations=[MLFLOW])


class DeploymentTriggerConfig(BaseParameters):
    """Deployment trigger config."""

    min_accuracy: float = 0.5
    
    
@step(enable_cache=False)
def dynamic_importer()->str:
    data = get_data_for_test()
    return data

@step
def deployment_trigger(accuracy:float, config:DeploymentTriggerConfig):
    """Implements a simple model deployment trigger that looks at the input model accuracy and decided if it is good enough to deploy or not."""
    
    return accuracy >= config.min_accuracy



class MLFlowDeploymentLoaderStepParameters(BaseParameters):
    """MLflow deployment getter parameters

    Attributes:
        pipeline_name: name of the pipeline that deployed the MLflow prediction
            server
        step_name: the name of the step that deployed the MLflow prediction
            server
        running: when this flag is set, the step only returns a running service
        model_name: the name of the model that is deployed
    """

    pipeline_name: str
    step_name: str
    running: bool = True
    

@step(enable_cache=False)
def prediction_service_loader(
    pipeline_name: str,pipeline_step_name: str,running: bool = True,model_name: str = "model",)-> MLFlowDeploymentService:
    
    mlflow_model_deployer_component = MLFlowModelDeployer.get_active_model_deployer()
    
    existing_services = mlflow_model_deployer_component.find_model_server(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        model_name=model_name,
        running=running,
    )
    
    if not existing_services:
        raise RuntimeError(
            f"No MLflow prediction service deployed by the "
            f"{pipeline_step_name} step in the {pipeline_name} "
            f"pipeline for the '{model_name}' model is currently "
            f"running."
        )
        
    print(existing_services)
    print(type(existing_services))
    return existing_services[0]


@step
def predictor(
    service:MLFlowDeploymentService, data: str,):
    """Predicts the output of the model deployed by the MLflow model deployer."""
    
    
    service.start(timeout=2000)
    data = json.loads(data)
    data.pop("columns")
    data.pop("index")
    
    columns_for_df = [
        'USMER',
        'MEDICAL_UNIT',
        'SEX',
        'PATIENT_TYPE',
        'INTUBED',
        'PNEUMONIA',
        'AGE',
        'PREGNANT',
        'DIABETES',
        'COPD',
        'ASTHMA',
        'INMSUPR',
        'HIPERTENSION',
        'OTHER_DISEASE',
        'CARDIOVASCULAR',
        'OBESITY',
        'RENAL_CHRONIC',
        'TOBACCO',
        'ICU'
    ]
    
    df = pd.DataFrame(data["data"], columns=columns_for_df)
    data = np.array(df)
    print("***********************************************************")
    
    try:
        prediction = service.predict(data)
        print("Done with the prediction!")
        print("***********************************************************")
        return prediction
    except Exception as e:
        print(e)
    


@pipeline(enable_cache=False, settings={"docker": docker_settings})
def continuous_deployment_pipeline(datapath: str,
                                   min_accuracy: float = 0.5, 
                                   workers: int = 1,
                                   timeout: int=DEFAULT_SERVICE_START_STOP_TIMEOUT,):
    # df_artifact = run(datapath)
    # df_cleaned = clean_data(df_artifact)
    # X,y = split_data(df_cleaned)
    # X_train, X_test, y_train, y_test = train_and_test_split(X,y)

    # model = train_model(X_train, y_train)
    # accuracy_score = model_eval(X_test, y_test, model)
    
    data = ingest_data(datapath)
    X,y = clean_data(data)
    
    X_train, X_test, y_train, y_test = split_data(X,y)
    model = model_train(X_train, y_train)
    
    acc = model_eval(X_test, y_test, model)
    
    deployment_decision = deployment_trigger(acc)
    mlflow_model_deployer_step(model=model, 
                               deploy_decision=deployment_decision, 
                               workers=workers, 
                               timeout=timeout,)
    
    
    
@pipeline(enable_cache=False, settings={"docker": docker_settings})
def inference_pipeline(pipeline_name: str, pipeline_step_name: str):
    # Link all the steps artifacts together
    batch_data = dynamic_importer()
    model_deployment_service = prediction_service_loader(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        running=False,
    )
    predictor(service=model_deployment_service, data=batch_data)