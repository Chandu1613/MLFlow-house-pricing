import mlflow
import pandas as pd
from typing import Dict

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


import matplotlib.pyplot as plt


def set_or_create_experiment(experiment_name: str) -> str:
    """
    Get or create an experiment.

    :param experiment_name: Name of the experiment.
    :return: Experiment ID.
    """

    try:
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    except:
        experiment_id = mlflow.create_experiment(experiment_name)
    finally:
        mlflow.set_experiment(experiment_name=experiment_name)

    return experiment_id

def get_regression_metrics(
    y_true: pd.DataFrame, y_pred: pd.DataFrame, prefix: str
) -> Dict[str, float]:
    """
    Log classification metrics.

    :param y_true: True labels.
    :param y_pred: Predicted labels.
    :param prefix: Prefix for the metric names.
    :return: Classification metrics.
    """
    metrics = {
        f"{prefix}_R2_Score": r2_score(y_true, y_pred),
        f"{prefix}_Mean_Absolute_Error": mean_absolute_error(y_true, y_pred),
        f"{prefix}_Mean_Squared_Error": mean_squared_error(y_true, y_pred),
    }

    return metrics

def register_model_with_client(model_name: str, run_id: str, artifact_path: str):
    """
    Register a model.

    :param model_name: Name of the model.
    :param run_id: Run ID.  
    :param artifact_path: Artifact path.

    :return: None.
    """
    client = mlflow.tracking.MlflowClient()
    client.create_registered_model(model_name)
    client.create_model_version(name=model_name, source=f"runs:/{run_id}/{artifact_path}")
        