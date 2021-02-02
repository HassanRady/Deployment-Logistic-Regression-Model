import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import re


def tolist(func):
    def wrapper(self, **kwargs):
        if not isinstance(kwargs['variables'], list):
            kwargs['variables'] = [kwargs['variables']]
        return func(self, **kwargs)
    return wrapper


class ValueReplacer(BaseEstimator, TransformerMixin):
    @tolist
    def __init__(self, variables=None):
        self.variables = variables

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        value = self.variables[0]
        new_value = self.variables[1]
        return X.replace(value, new_value)


class FirstCabinGetter(BaseEstimator, TransformerMixin):
    @tolist
    def __init__(self, variables=None):
        self.variables = variables

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        def get_first_cabin(x):
            try:
                x = x.split()[0]
            except:
                x = np.nan
            return x
        X[self.variables[0]] = X[self.variables[0]].apply(get_first_cabin)
        return X


class TitleGetter(BaseEstimator, TransformerMixin):
    @tolist
    def __init__(self, variables=None):
        self.variables = variables

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        name_var = self.variables[0]
        def get_title(passenger_name):
            if re.search('Mrs', passenger_name):
                x = 'Mrs'
            elif re.search('Mr', passenger_name):
                x = 'Mr'
            elif re.search('Miss', passenger_name):
                x = 'Miss'
            elif re.search('Master', passenger_name):
                x = 'Master'
            else:
                x = 'Other'
            return x
        X['title'] = X[name_var].apply(get_title)
        return X


class FloatConverter(BaseEstimator, TransformerMixin):
    @tolist
    def __init__(self, variables=None):
        self.variables = variables

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[self.variables[0]] = X[self.variables[0]].astype('float')
        return X


class StringConverter(BaseEstimator, TransformerMixin):
    @tolist
    def __init__(self, variables=None):
        self.variables = variables

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[self.variables[0]] = X[self.variables[0]].astype('str')
        return X


class FirstLetterGetter(BaseEstimator, TransformerMixin):
    @tolist
    def __init__(self, variables=None):
        self.variables = variables

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for var in self.variables:
            X[var] = X[var].str[0]
        return X


class ColumnsDropper(BaseEstimator, TransformerMixin):
    @tolist
    def __init__(self, variables=None):
        self.variables = variables

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X = X.drop(self.variables, axis=1)
        return X


class NullIndicator(BaseEstimator, TransformerMixin):
    @tolist
    def __init__(self, variables=None):
        self.variables = variables

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for var in self.variables:
            X[var+'_isnull'] = np.where(X[var].isnull(), 1, 0)
        return X


class NumericalImputer(BaseEstimator, TransformerMixin):
    @tolist
    def __init__(self, variables=None):
        self.variables = variables

    def fit(self, X, y=None):
        self._impute_values = {}
        for var in self.variables:
            self._impute_values[var] = X[var].median()
        return self

    def transform(self, X):
        X = X.copy()
        for var in self.variables:
            X[var].fillna(self._impute_values[var], inplace=True)
        return X


class CategoricalImputer(BaseEstimator, TransformerMixin):
    @tolist
    def __init__(self, variables=None):
        self.variables = variables

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for var in self.variables:
            X[var].fillna("missing", inplace=True)
        return X


class RaresRemover(BaseEstimator, TransformerMixin):
    @tolist
    def __init__(self, variables=None, tol=0.05):
        self.variables = variables
        self.tol = tol

    def fit(self, X, y=None):
        self._frequent_labels = {}
        for var in self.variables:
            tmp = X.groupby(var)[var].count() / len(X)
            self._frequent_labels[var] = tmp[tmp > self.tol].index
        return self

    def transform(self, X):
        X = X.copy()
        for var in self.variables:
            X[var] = np.where(X[var].isin(
                self._frequent_labels[var]), X[var], "rare")
        return X


class DummyEncoder(BaseEstimator, TransformerMixin):
    @tolist
    def __init__(self, variables=None):
        self.variables = variables

    def fit(self, X, y=None):
        self._dummies = pd.get_dummies(
            X[self.variables], drop_first=True).columns
        return self

    def transform(self, X):
        X = X.copy()
        X = pd.concat([X, pd.get_dummies(X[self.variables], drop_first=True)], axis=1).drop(
            self.variables, axis=1)
        missing_dummies = [
            var for var in self._dummies if var not in X.columns]
        if len(missing_dummies) == 0:
            print("No Missing Variables")
        else:
            for var in missing_dummies:
                X[var] = 0
        return X


class FeatureSelector(BaseEstimator, TransformerMixin):
    @tolist
    def __init__(self, variables=None):
        self.variables = variables

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X = X[self.variables]
        return X


if __name__ == '__main__':
    Pass
