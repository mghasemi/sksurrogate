"""
DataProcess: A DataFrame preprocessing module for ML
===========================================================

The module is designed to do some preprocessing task on a given DataFrame.
Currently, the module detects types of columns data e.g., numerical (integer, float), categorical, boolean, ordinal,
datetime, and labels.

The module performs ordinal encoding, one hot encoding, date range to float transformation, label encoding, and
missing data imputation.

To Do
----------
Processing of Text and Image data.

Code
----------
"""
import numpy
import pandas
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from category_encoders.one_hot import OneHotEncoder
from category_encoders.ordinal import OrdinalEncoder


class DateTime2Num(TransformerMixin, BaseEstimator):
    """
    Converts a date range data to numeric (float) between `lower` and `upper`.

    :param cols: default: `None`; column names for transformation.
    :param unit: default: 'day`; the time unit.
    :param lower: default: `0.0`; the lower bound of the transformation range.
    :param upper: default: `2.0`; the upper bound of the transformation range.
    """

    def __init__(self, cols=None, unit='day', lower=0., upper=2.):
        if unit not in ['day', 'hour', 'minute', 'second']:
            raise ValueError("`unit` must be either 'day', 'hour', 'minute', or 'second'.")
        self.cols = cols
        self.unit = unit
        self.lower = lower
        self.upper = upper
        self.coefs = dict()
        self.scale = 1.
        self.scale = numpy.timedelta64(1, 's')
        if self.unit == 'minute':
            self.scale = 60.
            self.scale = numpy.timedelta64(1, 'm')
        elif self.unit == 'hour':
            self.scale = 60 * 60.
            self.scale = numpy.timedelta64(1, 'h')
        elif self.unit == 'day':
            self.scale = 60 * 60 * 24.
            self.scale = numpy.timedelta64(1, 'D')

    def fit(self, X):
        """
        Fits and calculates the transformation parameters for date range transformation.

        :param X: Input data
        :return: `self`
        """
        if self.cols is None:
            return self
        n_length = self.upper - self.lower
        for clmn in self.cols:
            t_vec = X[clmn].dropna().values
            min_date, max_date = t_vec.min(), t_vec.max()
            # t_length = (max_date - min_date).seconds / self.scale
            t_length = (max_date - min_date) / self.scale
            trans_coef = n_length / t_length
            intercept = self.lower
            self.coefs[clmn] = (trans_coef, intercept, min_date)
        return self

    def transform(self, X):
        """
        Apply the calculated parameters to transform the datetime columns.

        :param X: the input data
        :return: transformed data
        """
        if self.cols is None:
            return X
        TX = X
        for clmn in self.cols:
            vect = numpy.array(
                [self.coefs[clmn][1] + self.coefs[clmn][0] * (_ - self.coefs[clmn][2]) / self.scale for _ in
                 TX[clmn].values])
            TX[clmn] = vect
        return TX

    def fit_transform(self, X, y=None, **fit_params):
        """
        Fits and transforms the input data.

        :param X: input data;
        :param y: target field; not required, but included to comply with scikit-learn's signature
        :param fit_params: included to comply with scikit-learn's signature
        :return: transformed DataFrame
        """
        self.fit(X)
        return self.transform(X)


class DataPreprocess(object):
    """
    This class performs a few reprocessing task on the input DataFrame to provide a ML ready dataset.

    :param df: the input raw DataFrame
    :param bin_threshold: float between 0. and 1.; when the proportion of unique values to the total number of entities
        falls below this number fo two valued columns, the type is deduced to be binary
    :param cat_threshold: when the proportion of unique values to the total number of entities
        falls below this number fo discrete columns, the type is deduced to be categorical or ordinal
    :param force_threshold: boolean, indicate whether to force the deduction based on the provided thresholds or not
    :param imputer: the imputer object; default `IterativeImputer` for missing values imputation
    :param force_impute: boolean; default `True` to indicate the desire for missing value imputation
    """

    def __init__(self, df, bin_threshold=.1, cat_threshold=.1, force_threshold=False, imputer=None, force_impute=True):
        self.df = df
        self.bin_threshold = bin_threshold
        self.cat_threshold = cat_threshold
        self.force_threshold = force_threshold
        self.columns = list(df.columns)
        self.values = dict()
        self.uniques = dict()
        self.types = dict()
        self.transforms = dict()
        self.mapping = dict()
        self.deduced_types = dict(float64=[], int64=[], datetime64=[], other=[],
                                  binary=[], categorical=[], label=[], obsolete=[])
        self.pivot_types = dict()
        self.processed_frame = dict()
        self.transformed_df = None
        self.steps = []
        for clmn in self.columns:
            self.values[clmn] = self.df[clmn].dropna().values
            self.uniques[clmn] = self.df[clmn].dropna().unique()
            self.types[clmn] = self.df[clmn].dtypes.name
        self.types_deduced = False
        self.encoded = False
        self.force_impute = force_impute
        if imputer is None:
            self.imputer = IterativeImputer()
        else:
            self.imputer = imputer

    def is_float(self, clmn):
        """
        Determines if the type of the data for the column `clmn` is float or not

        :param clmn: the name of the column
        :return: boolean
        """
        if self.types[clmn] == 'float64':
            return True
        return False

    def is_int(self, clmn):
        """
        Determines if the type of the data for the column `clmn` is integer or not

        :param clmn: the name of the column
        :return: boolean
        """
        if self.types[clmn] == 'int64':
            return True
        return False

    def is_object(self, clmn):
        """
        Determines if the type of the data for the column `clmn` is object (e.g., string, blob, etc.) or not

        :param clmn: the name of the column
        :return: boolean
        """
        if self.types[clmn] == 'object':
            return True
        return False

    def is_datetime(self, clmn):
        """
        Determines if the type of the data for the column `clmn` is datetime or not

        :param clmn: the name of the column
        :return: boolean
        """
        if 'datetime64' in self.types[clmn]:
            return True
        return False

    def is_obsolete(self, clmn):
        """
        Determines if the column is worth to be included in the analysis or not

        :param clmn: the name of the column
        :return: boolean; `True` if includes more than 1 value, `False` otherwise
        """
        if len(self.uniques[clmn]) < 2:
            return True
        return False

    def is_bin(self, clmn):
        """
        Determines if the type of the data for the column `clmn` is binary or not

        :param clmn: the name of the column
        :return: boolean
        """
        ratio = len(self.uniques[clmn]) / len(self.values[clmn])
        if self.force_threshold and (len(self.uniques[clmn]) == 2) and (ratio <= self.bin_threshold):
            return True
        if not self.force_threshold and (len(self.uniques[clmn]) == 2):
            return True
        return False

    def is_categorical(self, clmn):
        """
        Determines if the type of the data for the column `clmn` is categorical or not

        :param clmn: the name of the column
        :return: boolean
        """
        ratio = len(self.uniques[clmn]) / len(self.values[clmn])
        if self.force_threshold and (ratio <= self.cat_threshold):
            return True
        elif (ratio <= self.cat_threshold) and (self.types[clmn] == 'object'):
            return True
        else:
            return False

    def is_label(self, clmn):
        """
        Determines if the type of the data for the column `clmn` is label or not

        :param clmn: the name of the column
        :return: boolean
        """
        if (self.types[clmn] == 'object') and not self.is_categorical(clmn):
            return True
        return False

    def deduce_types(self):
        """
        Deduces the type of all columns.

        :return: `None`
        """
        for clmn in self.columns:
            if self.is_label(clmn):
                self.deduced_types['label'].append(clmn)
                self.pivot_types[clmn] = 'label'
            elif self.is_bin(clmn):
                self.deduced_types['binary'].append(clmn)
                self.pivot_types[clmn] = 'binary'
            elif self.is_categorical(clmn):
                self.deduced_types['categorical'].append(clmn)
                self.pivot_types[clmn] = 'categorical'
            elif self.is_datetime(clmn):
                self.deduced_types['datetime64'].append(clmn)
                self.pivot_types[clmn] = 'datetime64'
            elif self.is_object(clmn):
                self.deduced_types['other'].append(clmn)
                self.pivot_types[clmn] = 'other'
            elif self.is_int(clmn):
                self.deduced_types['int64'].append(clmn)
                self.pivot_types[clmn] = 'int64'
            elif self.is_float(clmn):
                self.deduced_types['float64'].append(clmn)
                self.pivot_types[clmn] = 'float64'
            elif self.is_obsolete(clmn):
                self.deduced_types['obsolete'].append(clmn)
                self.pivot_types[clmn] = 'obsolete'
        self.types_deduced = True

    def transform_label_bin(self, clmn):
        """
        Transforms the determined label and binary columns.

        :param clmn: the column's name
        :return: `None`
        """
        lbls = numpy.sort(self.uniques[clmn])
        self.transforms[clmn] = {_: lbls[_] for _ in range(lbls.shape[0])}
        self.mapping[clmn] = {lbls[_]: _ for _ in range(lbls.shape[0])}

    def encode(self):
        """
        Transforms the whole input DataFrame based on the deduced types and does missing data imputation if desired.
        The resulted DataFrame will be stored in `transformed_df`

        :return: Transformed and encoded dataframe
        """
        if not self.types_deduced:
            self.deduce_types()
        self.steps = []
        ohe = OneHotEncoder(cols=self.deduced_types['categorical'], drop_invariant=True, handle_missing='return_nan',
                            handle_unknown='return_nan')
        ordinal_columns = self.deduced_types['binary'] + self.deduced_types['label']
        for clmn in ordinal_columns:
            self.transform_label_bin(clmn)
        clmn_maps = [{'col': _, 'mapping': self.mapping[_]} for _ in
                     self.deduced_types['binary'] + self.deduced_types['label']]
        oe = OrdinalEncoder(cols=ordinal_columns, mapping=clmn_maps, handle_missing='return_nan',
                            handle_unknown='return_nan')
        dtn = DateTime2Num(cols=self.deduced_types['datetime64'])
        self.steps.append(('OneHot', ohe))
        self.steps.append(('Ordinal', oe))
        self.steps.append(('Date2Num', dtn))
        self.steps.append(('Impute', self.imputer))
        trans = Pipeline(self.steps)
        self.transformed_df = pandas.DataFrame(trans.fit_transform(self.df), columns=oe.feature_names)
        for clmn in ohe.feature_names:
            if clmn not in self.columns:
                if self.force_impute:
                    self.transformed_df[clmn] = self.transformed_df.apply(lambda x: int(round(x[clmn], 0)), axis=1)
                else:
                    self.transformed_df[clmn] = self.transformed_df.apply(lambda x: round(x[clmn], 0), axis=1)
            elif clmn in self.deduced_types['categorical'] + ordinal_columns:
                if self.force_impute:
                    self.transformed_df[clmn] = self.transformed_df.apply(lambda x: int(round(x[clmn], 0)), axis=1)
                else:
                    self.transformed_df[clmn] = self.transformed_df.apply(lambda x: round(x[clmn], 0), axis=1)
        self.encoded = True
        return self.transformed_df
