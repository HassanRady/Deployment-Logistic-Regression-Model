import numpy as np
import logging
from sklearn.model_selection import train_test_split

from logistic_regression_model import pipeline
from logistic_regression_model.processing.data_management import load_dataset, save_pipeline
from logistic_regression_model.config import config
from logistic_regression_model import __version__ as _version

_logger = logging.getLogger(__name__)


def train() -> None:
    data = load_dataset(config.DATA_FILE)

    X_train, X_test, y_train, y_test = train_test_split(
        data.drop(config.TARGET, axis=1), data[config.TARGET], test_size=0.2, random_state=config.SEED)

    pipeline.pipe.fit(X_train, y_train)
    _logger.info(f"Saving Pipeline: {_version}")
    save_pipeline(pipeline.pipe)

if __name__ == '__main__':
    train()
