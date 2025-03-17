from ml_package.package.feature.data_processing import get_feature_dataframe

if __name__ == "__main__":
    experiment_name = "house_pricing_classifier"
    run_name = "training_classifier"
    model_name = "registered_model"
    artifact_path = "model"

    df = get_feature_dataframe()
    print(df.head())