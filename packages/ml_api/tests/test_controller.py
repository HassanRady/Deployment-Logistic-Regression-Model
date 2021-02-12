from logistic_regression_model.config import config as model_config
from logistic_regression_model.processing.data_management import load_dataset
from logistic_regression_model import __version__ as _version
import json


def test_health_endpoint_returns_200(flask_test_client):
    # When
    response = flask_test_client.get('/health')

    # Then
    assert response.status_code == 200


def test_prediction_endpoint_returns_prediction(flask_test_client):
    test_data = load_dataset(model_config.DATA_FILE)
    post_json = test_data[0:1].to_json(orient='records')
    print(type(post_json))

    response = flask_test_client.post('/v1/predict/logistic',
                                      json=post_json)
    assert response.status_code == 200

    response_json = json.loads(response.data)
    prediction = response_json['predictions']
    response_version = response_json['version']
    assert prediction == 1
    assert response_version == _version


