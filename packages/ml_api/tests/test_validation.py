import json
from logistic_regression_model.config import config
from logistic_regression_model.processing.data_management import load_dataset

def test_prediction_endpoint_validation_200(flask_test_client):
    test_data = load_dataset(config.DATA_FILE)
    post_json = test_data.to_json(orient='records')

    response = flask_test_client.post('/v1/predict/logistic',
                                      json=json.loads(post_json))

    assert response.status_code == 200

    response_json = json.loads(response.data)
    assert len(response_json.get('predictions')) + len(response_json.get('errors')) == len(test_data)