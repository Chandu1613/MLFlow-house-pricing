from ml_package.package.feature.data_processing import get_feature_dataframe

from ml_package.package.ml_training.retrival import get_train_test_score_set
from ml_package.package.ml_training.train import train_model
from ml_package.package.ml_training.preprocessing_pipeline import get_pipeline

from ml_package.package.utils.utils import set_or_create_experiment
from ml_package.package.utils.utils import get_regression_metrics
from ml_package.package.utils.utils import register_model_with_client
import mlflow

if __name__ == "__main__":
    experiment_name = "house_pricing_regression"
    run_name = "training"
    model_name = "registered_model"
    artifact_path = "model"

    df = get_feature_dataframe()

    x_train, x_test, x_score, y_train, y_test, y_score = get_train_test_score_set(df)

    features = [f for f in x_train.columns if f not in ["id", "target", "MedHouseVal"]]

    pipeline = get_pipeline(numerical_features=features, categorical_features=[])

    experiment_id = set_or_create_experiment(experiment_name=experiment_name)

    run_id, model = train_model(pipeline=pipeline, run_name=run_name, model_name=model_name, artifact_path=artifact_path, x=x_train[features], y=y_train)

    y_pred = model.predict(x_test)

    regression_metrics = get_regression_metrics(y_true=y_test,y_pred=y_pred,prefix='test')

    # log performance metrics
    with mlflow.start_run(run_id=run_id):

        # log metrics
        mlflow.log_metrics(regression_metrics)

        # log params
        mlflow.log_params(model[-1].get_params())

        # log tags
        mlflow.set_tags({"type":"regression"})

        # log description
        mlflow.set_tag(
            "mlflow.note.content", "This is a regressor for the house pricing dataset"
        )