import math
import pytest
from logistic_regression_model.config import config
from logistic_regression_model.predict import predict as model_predict
from logistic_regression_model.processing.data_management import load_dataset


@pytest.mark.differential
def test_model_prediction_differential(*args, save_file='test_data_predictions.csv'):
    """
    This test compares the prediction result similarity of
    the current model with the previous model's results.
    """
    # Given
    previous_model_df = load_dataset('test_data_predictions.csv')
    previous_model_predictions = previous_model_df.predictions.values
    test_data = load_dataset(config.DATA_FILE)
    test_data = test_data[50:555]

    # When
    response = model_predict(test_data)
    current_model_predictions = response.get('predictions')

    assert len(previous_model_predictions) == len(current_model_predictions)

    for previous_value, current_value in zip(previous_model_predictions, current_model_predictions):
        previous_value = previous_value.item()
        current_value = current_value.item()

        assert math.isclose(previous_value, current_value, rel_tol=config.ACCEPTABLE_MODEL_DIFFERENCE)
