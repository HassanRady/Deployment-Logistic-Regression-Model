import pandas as pd
from logistic_regression_model.predict import predict as model_predict
from logistic_regression_model.processing.data_management import load_dataset
from logistic_regression_model.config import config as model_config
from api import config


def capture_predictions():
    filename = "test_data_predictions"
    test_data = load_dataset(model_config.DATA_FILE)
    test_data = test_data[50:555]

    predictions = model_predict(test_data)
    predictions_df = pd.DataFrame(predictions)

    predictions_df.to_csv(
        f"{config.PACKAGE_ROOT.parent}/{filename}")



if __name__ == '__main__':
    capture_predictions()