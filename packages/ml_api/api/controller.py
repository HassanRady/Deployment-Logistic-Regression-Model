from flask import Blueprint, request, jsonify
from logistic_regression_model.predict import predict as model_predict
from api.config import get_logger

_logger = get_logger(logger_name=__name__)


prediction_app = Blueprint('prediction_app', __name__)


@prediction_app.route('/health', methods=['GET'])
def health():
    if request.method == 'GET':
        _logger.info('health status OK')
        return 'ok'


@prediction_app.route('/v1/predict/logistic', methods=['POST'])
def predict():
    if request.method == 'POST':
        json_data = request.get_json()
        _logger.info(f'Inputs: {json_data}')

        result = model_predict(data=json_data)
        _logger.info(f'Outputs: {result}')

        predictions = int(result.get('predictions')[0])
        version = result.get('version')

        return jsonify({'predictions': predictions,
                        'version': version})
