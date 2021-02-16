import pandas as pd
import joblib
from logistic_regression_model.config import config
from logistic_regression_model import __version__ as _version
import logging
from sklearn.pipeline import Pipeline
import os
import typing as t

_logger = logging.getLogger(__name__)


def load_dataset(filename: str) -> pd.DataFrame:
    _dataset = pd.read_csv(f"{config.DATASET_DIR}/{filename}")
    return _dataset


def load_pipeline(filename: str) -> Pipeline:
    file_path = os.path.join(config.TRAINED_MODEL_DIR, filename)
    pipeline = joblib.load(file_path)
    return pipeline


def save_pipeline(pipeline: Pipeline) -> None:
    save_file_name = f"{config.PIPELINE_SAVE_FILE}{_version}.pkl"
    save_path = os.path.join(config.TRAINED_MODEL_DIR, save_file_name)

    remove_old_pipelines(files_to_keep=[save_file_name])
    joblib.dump(pipeline, save_path)
    _logger.info(f"saved pipeline: {save_file_name}")


def remove_old_pipelines(*args, files_to_keep: t.List[str]) -> None:
    """
    Remove old model pipelines.
    This is to ensure there is a simple one-to-one
    mapping between the package version and the model
    version to be imported and used by other applications.
    """
    keep = files_to_keep + ["__init__.py"]
    for model_file in config.TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in keep:
            model_file.unlink()