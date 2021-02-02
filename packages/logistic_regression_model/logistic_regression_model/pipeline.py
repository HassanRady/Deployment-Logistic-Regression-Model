from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from logistic_regression_model.config import config
from logistic_regression_model.processing import preprocessors as pp

C = 5e-4

pipe = Pipeline([
    ("replace_with_nan", pp.ValueReplacer(variables=config.REPLACE_VALUE_WITH)),
    ("get_first_cabin", pp.FirstCabinGetter(variables=config.CABIN_VAR)),
    ("get_title", pp.TitleGetter(variables=config.NAME_VAR)),
    ("to_float", pp.FloatConverter(variables=config.NUMERICALS_TO_FLOAT)),
    ("get_first_letter", pp.FirstLetterGetter(variables=config.CABIN_VAR)),
    ("drop_columns", pp.ColumnsDropper(variables=config.COLUMNS_TO_DROP)),
    ("indicate_nulls", pp.NullIndicator(variables=config.NUMERICALS_TO_IMPUTE)),
    ("impute_numerics", pp.NumericalImputer(variables=config.NUMERICALS_TO_IMPUTE)),
    ("impute_categorics", pp.CategoricalImputer(variables=config.CATEGORICAL_TO_IMPUTE)),
    ("remove_rares", pp.RaresRemover(variables=config.CATEGORICAL_TO_IMPUTE)),
    ("encode_categorics", pp.DummyEncoder(variables=config.CATEGORICAL_TO_ENCODE)),
    ("order_features", pp.FeatureSelector(variables=config.FEATURES)),
    ('scale', StandardScaler()),
    ("model", LogisticRegression(C=C, n_jobs=-1, random_state=config.SEED))
])