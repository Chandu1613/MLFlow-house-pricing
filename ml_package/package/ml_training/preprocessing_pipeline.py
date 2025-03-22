from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor

from typing import List


def get_pipeline(
    numerical_features: List[str], categorical_features: List[str]
) -> Pipeline:
    """
    Get sklearn pipeline for regression with scaling.

    :param numerical_features: List of numerical features.
    :param categorical_features: List of categorical features.
    :return: Sklearn pipeline.
    """
    transformer = ColumnTransformer(
        [
            (
                "numerical_pipeline",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),  # Scaling numerical features
                    ]
                ),
                numerical_features,
            ),
            (
                "categorical_pipeline",
                Pipeline(
                    [
                        ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    pipeline = Pipeline(
        [("transformer", transformer), ("regressor", RandomForestRegressor())]
    )

    return pipeline