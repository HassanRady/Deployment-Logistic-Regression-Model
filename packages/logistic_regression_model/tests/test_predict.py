from logistic_regression_model.config import config
from logistic_regression_model.predict import predict
from logistic_regression_model.processing.data_management import load_dataset
import numpy as np

def test_predict():
    # Given
    test_data = load_dataset(config.DATA_FILE)
    original_data_length = len(test_data)
    multiple_test_json = test_data   #.to_json(orient='records')

    # When
    subject = predict(multiple_test_json)

    # Then
    assert subject is not None
    assert len(subject.get('predictions')) == 1309

    # We expect some rows to be filtered out
    # assert len(subject.get('predictions')) != original_data_length


def test_make_single_prediction():
    # Given
    test_data = load_dataset(config.DATA_FILE)
    single_test_json = test_data[0:1]

    # When
    subject = predict(single_test_json)

    # Then
    assert subject is not None
    assert isinstance(subject.get('predictions')[0], np.int64)
    assert subject.get('predictions')[0] == [1]