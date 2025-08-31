import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import RobustScaler


class DataFrameImputer(TransformerMixin, BaseEstimator):
    """
    A class to impute missing values in a Pandas DataFrame using a combination of median, knn, and most frequent
    imputers on specified columns.

    Parameters:
    -----------
    median_cols : list of str, optional (default=None)
        Columns to impute missing values using the median imputer.
    knn_cols : list of str, optional (default=None)
        Columns to impute missing values using the KNN imputer.
    freq_cols : list of str, optional (default=None)
        Columns to impute missing values using the most frequent imputer.
    const_cols : dict of {column_name: constant_value} pairs, optional (default=None)
        Columns to impute missing values using a constant value.

    Returns:
    --------
    X_imputed : pandas.DataFrame
        A DataFrame with imputed missing values.
    """
    def __init__(self, median_cols=None, knn_cols=None, freq_cols=None, const_cols=None, fill_const=0):
        self.median_cols = median_cols
        self.knn_cols = knn_cols
        self.freq_cols = freq_cols
        self.const_cols = const_cols
        self.fill_const = fill_const

    def fit(self, X, y=None):
        self.median_imputer = SimpleImputer(strategy='median')
        self.knn_imputer = KNNImputer()
        self.freq_imputer = SimpleImputer(strategy='most_frequent')
        self.const_imputer = SimpleImputer(strategy='constant', fill_value=self.fill_const)

        if self.median_cols is not None:
            self.median_imputer.fit(X[self.median_cols])
        if self.knn_cols is not None:
            self.knn_imputer.fit(X[self.knn_cols])
        if self.freq_cols is not None:
            self.freq_imputer.fit(X[self.freq_cols])
        if self.const_cols is not None:
            self.const_imputer.fit(X[self.const_cols])

        return self

    def transform(self, X):
        X_imputed = X.copy()
        if self.median_cols is not None:
            X_median = pd.DataFrame(self.median_imputer.transform(X[self.median_cols]),
                                    columns=self.median_cols, index=X.index)
            X_imputed = pd.concat([X_imputed.drop(self.median_cols, axis=1), X_median], axis=1)
        if self.knn_cols is not None:
            X_knn = pd.DataFrame(self.knn_imputer.transform(X[self.knn_cols]),
                                 columns=self.knn_cols, index=X.index)
            X_imputed = pd.concat([X_imputed.drop(self.knn_cols, axis=1), X_knn], axis=1)
        if self.freq_cols is not None:
            X_freq = pd.DataFrame(self.freq_imputer.transform(X[self.freq_cols]),
                                  columns=self.freq_cols, index=X.index)
            X_imputed = pd.concat([X_imputed.drop(self.freq_cols, axis=1), X_freq], axis=1)
        if self.const_cols is not None:
            X_const = pd.DataFrame(self.const_imputer.transform(X[self.const_cols]),
                                  columns=self.const_cols, index=X.index)
            X_imputed = pd.concat([X_imputed.drop(self.const_cols, axis=1), X_const], axis=1)
        return X_imputed

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)
    


class OutlierThresholdTransformer(BaseEstimator, TransformerMixin):
    """
    A custom scikit-learn transformer for outlier treatment using the IQR method.

    This transformer detects outliers in numerical columns based on the Interquartile
    Range (IQR) and caps them at the calculated lower and upper thresholds.
    Specifically, values below `Q1 - 1.5*IQR` are set to the lower limit, and values
    above `Q3 + 1.5*IQR` are set to the upper limit.

    Parameters
    ----------
    column : list of str
        List of column names to apply outlier treatment on.
    q1 : float, default=0.25
        The quantile to use for the lower quartile (25th percentile).
    q3 : float, default=0.75
        The quantile to use for the upper quartile (75th percentile).

    Attributes
    ----------
    None

    Methods
    -------
    fit(X, y=None):
        Does nothing and returns self. Added for pipeline compatibility.
    transform(X):
        Returns a copy of X with outliers in specified columns capped.
    fit_transform(X, y=None):
        Equivalent to calling fit followed by transform.
    """
    def __init__(self, column, q1=0.25, q3=0.75):
        self.column = column
        self.q1 = q1
        self.q3 = q3
    def outlier_threshhold(self, dataframe, column):
        Q1 = dataframe[column].quantile(self.q1)
        Q3 = dataframe[column].quantile(self.q3)
        iqr = Q3 - Q1
        up_limit = Q3 + 1.5 * iqr
        low_limit = Q1 - 1.5 * iqr
        return low_limit, up_limit

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        for col in self.column:
            low_limit, up_limit = self.outlier_threshhold(X_copy, col)
            X_copy.loc[(X_copy[col] < low_limit), col] = low_limit
            X_copy.loc[(X_copy[col] > up_limit), col] = up_limit
        return X_copy

    def fit_transform(self, X, y=None):
        return self.transform(X)
    


class CustomOneHotEncoder(BaseEstimator, TransformerMixin):
    """
    A transformer class to apply one-hot encoding to specified columns in a Pandas DataFrame.
    Ensures consistent output columns regardless of which categories appear in transform data.

    Parameters
    ----------
    columns : list
        A list of column names to encode.

    Returns
    -------
    pandas.DataFrame
        A new DataFrame with the specified columns one-hot encoded.
    """

    def __init__(self, columns=None):
        self.columns = columns
        self.unique_values = {}
        self.feature_names_ = None

    def fit(self, X, y=None):
        if self.columns is None:
            self.columns = X.columns.tolist()
        # Store unique values from training data only
        self.unique_values = {col: sorted(X[col].dropna().unique()) for col in self.columns}
        self.feature_names_ = self._get_feature_names()
        return self

    def _get_feature_names(self):
        feature_names = []
        for col in self.columns:
            for value in self.unique_values[col]:
                feature_names.append(f"{col}_{value}")
        return feature_names

    def transform(self, X):
        # Create a copy to avoid modifying the original
        X_copy = X.copy()

        # Create DataFrame for one-hot encoded features
        X_transformed = pd.DataFrame(index=X.index)

        for col in self.columns:
            # Create one-hot columns for ALL categories seen during fit
            for value in self.unique_values[col]:
                # Check if the column exists and create the one-hot encoding
                if col in X_copy.columns:
                    X_transformed[f"{col}_{value}"] = (X_copy[col] == value).astype(int)
                else:
                    # If the column doesn't exist in transform data, fill with zeros
                    X_transformed[f"{col}_{value}"] = 0

        # Remove original categorical columns
        X_remaining = X_copy.drop(columns=[col for col in self.columns if col in X_copy.columns], errors="ignore")

        # Combine remaining features with one-hot encoded features
        # Ensure all expected one-hot columns exist (fill missing with 0)
        for feature_name in self.feature_names_:
            if feature_name not in X_transformed.columns:
                X_transformed[feature_name] = 0

        # Concatenate remaining columns with one-hot encoded columns
        result = pd.concat([X_remaining, X_transformed[self.feature_names_]], axis=1)

        return result

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)



class LogTransform(BaseEstimator, TransformerMixin):
    """
    A transformer class to apply a log transform to a specified column in a Pandas DataFrame.

    Parameters
    ----------
    columns : str
        The name of the column to apply the log transform to.
    domain_shift : float
        The value to be added to the column before applying the log transform.

    return
    ------
        transformed feature
    """
    def __init__(self, columns, domain_shift=1.0):
        self.columns = columns
        self.domain_shift = domain_shift

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy[self.columns] = np.log(X_copy[self.columns] + self.domain_shift)
        return X_copy

    def fit_transform(self, X, y=None):
        return self.transform(X)
    


class RobustScaleTransform(BaseEstimator, TransformerMixin):
    """
    A transformer class to apply standard scaling to specified columns in a Pandas DataFrame.

    Parameters
    ----------
    cols : list of str
        The names of the columns to apply standard scaling to.
    """
    def __init__(self, cols):
        self.cols = cols
        self.scaler_ = None

    def fit(self, X, y=None):
        self.scaler_ = RobustScaler().fit(X.loc[:, self.cols])
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy.loc[:, self.cols] = self.scaler_.transform(X_copy.loc[:, self.cols])
        return X_copy

    def fit_transform(self, X, y=None):
        self.scaler_ = RobustScaler().fit(X.loc[:, self.cols])
        return self.transform(X)
    


class EnsemblePredictor(TransformerMixin, BaseEstimator):
    def __init__(self, models_weights=None):
        self.models_weights = models_weights
        self.feature_names_in_ = None  # Store expected feature names

    def fit(self, X, y=None):
        # Store the feature names from training data
        if hasattr(X, 'columns'):
            self.feature_names_in_ = list(X.columns)
        return self

    def transform(self, X):
        return self.predict(X).reshape(-1, 1)  # Reshape for pipeline compatibility

    def predict(self, X):
        if self.models_weights is None:
            raise ValueError("models_weights must be provided")

        # Ensure X has the same columns as training data
        if hasattr(X, 'columns') and self.feature_names_in_ is not None:
            # Add missing columns with zeros
            for col in self.feature_names_in_:
                if col not in X.columns:
                    X[col] = 0
            # Reorder columns to match training order
            X = X[self.feature_names_in_]

        predictions = []
        total_weight = 0
        for model, weight in self.models_weights:
            pred = model.predict(X)
            predictions.append(pred * weight)
            total_weight += weight

        return np.sum(predictions, axis=0) / total_weight

    def set_models_weights(self, models_weights):
        """Set models and weights after initialization"""
        self.models_weights = models_weights
        return self
    

