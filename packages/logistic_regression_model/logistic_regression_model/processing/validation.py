from logistic_regression_model.config import config
import pandas as pd

def validate_without_nulls(data: pd.DataFrame) -> pd.DataFrame:
    """If there is a null value drop it"""
    
    _data = data.copy()

    if config.NUMERICAL_NA_NOT_ALLOWED :
        if _data[config.NUMERICAL_NA_NOT_ALLOWED].isnull().any().any():
            _data = _data.dropna(axis=0, subset=config.NUMERICAL_NA_NOT_ALLOWED)

    if config.CATEGORICAL_NA_NOT_ALLOWED:
        if _data[config.CATEGORICAL_NA_NOT_ALLOWED].isnull().any().any():
            _data = _data.dropna(axis=0, subset=config.CATEGORICAL_NA_NOT_ALLOWED)

    return _data