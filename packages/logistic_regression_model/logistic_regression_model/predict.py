import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

from logistic_regression_model.config import config
from logistic_regression_model.processing.data_management import load_dataset, load_pipeline
from  logistic_regression_model.processing.validation import validate_without_nulls
from logistic_regression_model import __version__ as _version

_logger = logging.getLogger(__name__)

pipeline_file_name = f"{config.PIPELINE_SAVE_FILE}{_version}.pkl"
pipe = load_pipeline(pipeline_file_name)

def getScore():
    data = load_dataset(config.DATA_FILE)

    # X_train, X_test, y_train, y_test = train_test_split(
    #     data.drop(config.TARGET, axis=1), data[config.TARGET], test_size=0.2, random_state=config.SEED)

    X_test = data.drop(config.TARGET, axis=1) 
    y_test = data[config.TARGET]

    _logger.info(f"Scoring with: {_version}")
    preds = pipe.predict(X_test)
    preds_proba = pipe.predict_proba(X_test)[:, 1]

    return preds, preds_proba, y_test

def predict(data):
    data = pd.DataFrame(data)

    data = validate_without_nulls(data)
    preds = pipe.predict(data)
    results = {"predictions": preds, "version": _version}

    _logger.info(f"Making predictions with model version: {_version}")
    # _logger.info(f"results: {results['predictions']}")

    return results

if __name__ == '__main__':
    preds, preds_proba, y_test = getScore()
    print(preds)
    print(f"test Accuracy: {accuracy_score(y_test, preds)}")
    print(f"test roc-auc: {roc_auc_score(y_test, preds_proba)}")
