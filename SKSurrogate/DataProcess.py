"""
DataProcess: A DataFrame preprocessing module for ML
===========================================================

The module is designed to do some preprocessing task on a given DataFrame.
Currently, the module detects types of columns data e.g., numerical (integer, float), categorical, boolean, ordinal,
datetime, and labels. To impose a text column, the type of the column should be declared before passing the DataFrame.
This can be done by a simple code such as `df.astype({'txt_column':pandas.StringDtype()})`

The module performs ordinal encoding, one hot encoding, date range to float transformation, label encoding, and
missing data imputation.

To Do
----------
Processing of Image data.

Code
----------
"""
import numpy as np

import numpy
import pandas
import pandas as pd
from copy import copy
import re
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.feature_extraction.text import TfidfVectorizer


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


def _datetime_frame_to_epoch_seconds(X):
    transformed = pandas.DataFrame(index=X.index)
    for column in X.columns:
        values = pandas.to_datetime(X[column], errors="coerce")
        seconds = values.astype("int64").astype(float) / 1e9
        seconds[values.isna()] = numpy.nan
        transformed[column] = seconds
    return transformed


def _flatten_text_column(X):
    return pandas.DataFrame(X).iloc[:, 0].fillna("").astype(str).to_numpy()


def _is_declared_text_dtype(dtype) -> bool:
    """Whether ``dtype`` marks an explicitly declared text column.

    pandas >= 3 infers plain string columns as a first-class ``StringDtype``
    whose missing value is ``nan``, so an ``isinstance`` check alone can no
    longer distinguish user-declared text from ordinary categorical strings.
    Declared text columns (e.g. ``astype("string")`` or ``pandas.StringDtype()``)
    carry :data:`pandas.NA` as their missing value on both pandas 2.x and 3.x;
    only those receive TF-IDF vectorisation, while inferred string columns fall
    through to one-hot encoding.
    """
    return isinstance(dtype, pandas.StringDtype) and dtype.na_value is pandas.NA


class FoldLocalDataPreprocessor(TransformerMixin, BaseEstimator):
    """Fit DataFrame preprocessing within the current training partition.

    Use this transformer inside an sklearn ``Pipeline`` so every cross-
    validation fold learns its own imputations, category vocabulary, and text
    vocabulary. Numeric columns receive median imputation, booleans and
    non-text object/category columns receive imputation and one-hot encoding,
    datetimes are converted to epoch seconds and imputed. Explicitly declared
    text columns (``astype("string")`` / ``pandas.StringDtype()``) receive
    per-column TF-IDF features; on pandas >= 3, plain string columns are
    inferred as a first-class string dtype and fall back to one-hot encoding
    like object columns.

    The existing :class:`DataPreprocess` class remains available for eager
    whole-frame transformations; this transformer is the leakage-safe choice
    when preprocessing is part of model selection or cross-validation.
    """

    def __init__(self, scale_numeric=False):
        self.scale_numeric = scale_numeric

    def fit(self, X, y=None):
        from sklearn.compose import ColumnTransformer
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

        if not isinstance(X, pandas.DataFrame):
            raise TypeError("FoldLocalDataPreprocessor requires a pandas DataFrame")
        if X.columns.has_duplicates:
            raise ValueError("FoldLocalDataPreprocessor requires unique column names")
        if X.shape[1] == 0:
            raise ValueError("FoldLocalDataPreprocessor requires at least one feature")

        self._input_columns = list(X.columns)
        self.n_features_in_ = X.shape[1]
        if hasattr(self, "feature_names_in_"):
            del self.feature_names_in_
        if all(isinstance(column, str) for column in self._input_columns):
            self.feature_names_in_ = numpy.asarray(self._input_columns, dtype=object)

        numeric_columns = []
        categorical_columns = []
        datetime_columns = []
        text_columns = []
        for index, column in enumerate(X.columns):
            dtype = X[column].dtype
            if pandas.api.types.is_datetime64_any_dtype(dtype):
                datetime_columns.append(index)
            elif _is_declared_text_dtype(dtype):
                text_columns.append(index)
            elif isinstance(dtype, pandas.StringDtype):
                # pandas >= 3 infers plain string columns as a first-class
                # StringDtype; only explicitly declared ones are text.
                categorical_columns.append(index)
            elif pandas.api.types.is_bool_dtype(dtype):
                categorical_columns.append(index)
            elif pandas.api.types.is_numeric_dtype(dtype):
                numeric_columns.append(index)
            else:
                categorical_columns.append(index)

        transformers = []
        if numeric_columns:
            numeric_steps = [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True))
            ]
            if self.scale_numeric:
                numeric_steps.append(("scaler", StandardScaler()))
            transformers.append(
                (
                    "numeric",
                    Pipeline(numeric_steps),
                    numeric_columns,
                )
            )
        if categorical_columns:
            transformers.append(
                (
                    "categorical",
                    Pipeline(
                        [
                            (
                                "imputer",
                                SimpleImputer(
                                    strategy="constant",
                                    fill_value="__MISSING__",
                                    keep_empty_features=True,
                                ),
                            ),
                            ("onehot", OneHotEncoder(handle_unknown="ignore")),
                        ]
                    ),
                    categorical_columns,
                )
            )
        if datetime_columns:
            datetime_steps = [
                (
                    "epoch_seconds",
                    FunctionTransformer(
                        _datetime_frame_to_epoch_seconds,
                        feature_names_out="one-to-one",
                    ),
                ),
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ]
            if self.scale_numeric:
                datetime_steps.append(("scaler", StandardScaler()))
            transformers.append(
                (
                    "datetime",
                    Pipeline(datetime_steps),
                    datetime_columns,
                )
            )
        for index in text_columns:
            transformers.append(
                (
                    "text_%d" % index,
                    Pipeline(
                        [
                            (
                                "flatten",
                                FunctionTransformer(
                                    _flatten_text_column,
                                    feature_names_out="one-to-one",
                                ),
                            ),
                            ("tfidf", TfidfVectorizer(min_df=1)),
                        ]
                    ),
                    [index],
                )
            )

        self.transformer_ = ColumnTransformer(
            transformers,
            remainder="drop",
            verbose_feature_names_out=True,
        )
        self.transformer_.fit(X, y)
        return self

    def transform(self, X):
        if not hasattr(self, "transformer_"):
            raise RuntimeError("FoldLocalDataPreprocessor has not been fitted")
        if not isinstance(X, pandas.DataFrame):
            raise TypeError("FoldLocalDataPreprocessor requires a pandas DataFrame")
        if X.columns.has_duplicates:
            raise ValueError("FoldLocalDataPreprocessor requires unique column names")
        missing = [column for column in self._input_columns if column not in X.columns]
        extra = [column for column in X.columns if column not in self._input_columns]
        if missing or extra:
            raise ValueError(
                "Input columns do not match fitted columns; missing=%r, extra=%r"
                % (missing, extra)
            )
        return self.transformer_.transform(X.loc[:, self._input_columns])

    def get_feature_names_out(self, input_features=None):
        if not hasattr(self, "transformer_"):
            raise RuntimeError("FoldLocalDataPreprocessor has not been fitted")
        return self.transformer_.get_feature_names_out()


class ArrayAsFrame(TransformerMixin, BaseEstimator):
    """Rebuild a DataFrame from positional rows for a DataFrame-first pipeline.

    :class:`FoldLocalDataPreprocessor` needs column names and dtypes, so a
    pipeline that starts with it only accepts DataFrames. Several consumers
    legitimately hold positional arrays instead — partial dependence and
    accumulated local effects vary one column of a value matrix, learning-curve
    helpers and legacy trackers predict from ``X.to_numpy()`` — and used to
    fail with ``TypeError: FoldLocalDataPreprocessor requires a pandas
    DataFrame`` once experiments were fitted on mixed-type data.

    Prefixing such a pipeline with this transformer restores the array contract:
    DataFrames pass through untouched, arrays are labelled with the fitted
    column names in the order the preprocessor expects.

    :param columns: Feature names, in training order. When omitted the names of
        a DataFrame seen in :meth:`fit` are used.
    """

    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        if self.columns is not None:
            self.columns_ = list(self.columns)
        elif isinstance(X, pandas.DataFrame):
            self.columns_ = list(X.columns)
        else:
            self.columns_ = ["x%d" % index for index in range(np.asarray(X).shape[1])]
        self.n_features_in_ = len(self.columns_)
        return self

    def _column_names(self):
        names = getattr(self, "columns_", None)
        if names is None and self.columns is not None:
            names = list(self.columns)
        if names is None:
            raise RuntimeError(
                "ArrayAsFrame has no column names; pass columns= or fit it on a DataFrame"
            )
        return names

    def transform(self, X):
        if isinstance(X, pandas.DataFrame):
            return X
        names = self._column_names()
        array = numpy.asarray(X)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2 or array.shape[1] != len(names):
            raise ValueError(
                "ArrayAsFrame expected %d feature column(s); got shape %r"
                % (len(names), array.shape)
            )
        return pandas.DataFrame(array, columns=pandas.Index(names))


def frame_tolerant(estimator, columns=None):
    """Make a DataFrame-first pipeline accept positional arrays as well.

    Returns ``estimator`` unchanged unless its first step is a
    :class:`FoldLocalDataPreprocessor`, in which case the pipeline is prefixed
    with :class:`ArrayAsFrame`. Wrapping is idempotent: a pipeline whose first
    step is already an ``ArrayAsFrame`` is returned as-is.

    :param estimator: A fitted sklearn pipeline, or anything else (returned
        unchanged).
    :param columns: Feature names in training order; defaults to the names the
        preprocessor recorded.
    """
    steps = getattr(estimator, "steps", None)
    if not steps:
        return estimator
    _, first_step = steps[0]
    if isinstance(first_step, ArrayAsFrame):
        return estimator
    if not isinstance(first_step, FoldLocalDataPreprocessor):
        return estimator
    if columns is None:
        columns = getattr(first_step, "_input_columns", None)
        if columns is None:
            feature_names = getattr(first_step, "feature_names_in_", None)
            columns = None if feature_names is None else list(feature_names)
    return Pipeline([("as_frame", ArrayAsFrame(columns)), *steps])

def preprocesses_frames(estimator):
    """Whether ``estimator`` already turns a raw frame into model input.

    True when the first step of a pipeline is a
    :class:`FoldLocalDataPreprocessor`, directly or behind the
    :class:`ArrayAsFrame` adapter. Nested cross-validation uses this to decide
    whether a candidate still needs an encoder in front of it: a bundle trained
    from a preset already carries one, and stacking a second encoder on top fed
    the pipeline an already-encoded matrix.
    """
    steps = getattr(estimator, "steps", None)
    if not steps:
        return False
    first_step = steps[0][1]
    if isinstance(first_step, ArrayAsFrame):
        steps = steps[1:]
        if not steps:
            return False
        first_step = steps[0][1]
    return isinstance(first_step, FoldLocalDataPreprocessor)


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

    .. note::
        This is a one-shot, whole-frame encoder: types are deduced and missing values imputed on the entire
        frame, which leaks information across cross-validation folds. For scikit-learn pipelines and
        cross-validation use :class:`FoldLocalDataPreprocessor`, which has ``fit``/``transform``.
    """

    def __init__(self, df, bin_threshold=.1, cat_threshold=.1, force_threshold=False, imputer=None, force_impute=True,
                 text_processor=None):
        self.df = copy(df)
        self.bin_threshold = bin_threshold
        self.cat_threshold = cat_threshold
        self.force_threshold = force_threshold
        self.columns = list(df.columns)
        self.values = dict()
        self.uniques = dict()
        self.types = dict()
        self.transforms = dict()
        self.mapping = dict()
        self.deduced_types = dict(float64=[], int64=[], datetime64=[], other=[], text=[],
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
        if text_processor is None:
            self.txt_prcsr = TfidfVectorizer(sublinear_tf=True, max_features=1600, min_df=5, norm='l2',
                                             ngram_range=(1, 3), stop_words='english')
        else:
            self.txt_prcsr = text_processor

    @staticmethod
    def numeric(val, typ='float64'):
        trans_val = re.sub(r"[^\d\.]", "", str(val))
        if typ == 'float64':
            return float(trans_val)
        elif typ == 'int64':
            return int(trans_val)

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

    def is_numeric_string(self, clmn):
        """Return whether an object column contains only numeric values and nulls."""
        if self.types[clmn] != 'object':
            return False
        values = pandas.to_numeric(self.df[clmn].dropna(), errors='coerce')
        return len(values) > 0 and values.notna().all()

    def is_datetime(self, clmn):
        """
        Determines if the type of the data for the column `clmn` is datetime or not

        :param clmn: the name of the column
        :return: boolean
        """
        if 'datetime64' in self.types[clmn]:
            return True
        return False

    def is_text(self, clmn):
        """
        Determine if the type of the data for the column `clmn` is string/text or not. A string column is of type
        `pandas.StringDtype()`.

        :param clmn: the name of the column
        :return: boolean
        """
        if 'string' in self.types[clmn]:
            self.df[clmn].fillna("", inplace=True)
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

    def set_type(self, clmn, typ):
        if typ not in self.deduced_types:
            raise "Unknown type."
        self.deduced_types[typ].append(clmn)
        self.pivot_types[clmn] = typ

    def deduce_types(self):
        """
        Deduces the type of all columns.

        :return: `None`
        """
        for clmn in self.columns:
            if clmn in self.pivot_types:
                continue
            if self.is_numeric_string(clmn):
                values = pandas.to_numeric(self.df[clmn].dropna(), errors='coerce')
                typ = 'int64' if (values % 1 == 0).all() else 'float64'
                self.deduced_types[typ].append(clmn)
                self.pivot_types[clmn] = typ
            elif self.is_label(clmn):
                self.deduced_types['label'].append(clmn)
                self.pivot_types[clmn] = 'label'
            elif self.is_text(clmn):
                self.deduced_types['text'].append(clmn)
                self.pivot_types[clmn] = 'text'
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
        txt_clmn_df = None
        org_df = copy(self.df)
        for clmn in self.deduced_types['int64']:
            values = org_df[clmn].astype(str).str.replace(r"[^\d\.]", "", regex=True)
            org_df[clmn] = pandas.to_numeric(values, errors='coerce')
        for clmn in self.deduced_types['float64']:
            values = org_df[clmn].astype(str).str.replace(r"[^\d\.]", "", regex=True)
            org_df[clmn] = pandas.to_numeric(values, errors='coerce')
        if self.deduced_types['text']:
            txt_clmn_df = self.df[self.deduced_types['text']]
            org_df.drop(self.deduced_types['text'], axis=1, inplace=True)
        ordinal_columns = self.deduced_types['binary'] + self.deduced_types['label']
        for clmn in ordinal_columns:
            self.transform_label_bin(clmn)

        working_df = org_df.copy()
        for clmn in ordinal_columns:
            working_df[clmn] = working_df[clmn].map(self.mapping[clmn]).astype(float)

        for clmn in self.deduced_types['categorical']:
            dummy = pandas.get_dummies(
                working_df[clmn].astype(object).fillna("__MISSING__"),
                prefix=str(clmn),
                prefix_sep="_",
                dummy_na=False,
            )
            working_df = pandas.concat([working_df.drop(columns=[clmn]), dummy], axis=1)

        for clmn in self.deduced_types['datetime64']:
            working_df[clmn] = pandas.to_datetime(working_df[clmn], errors='coerce')
            working_df[clmn] = (
                (working_df[clmn] - pandas.Timestamp("1970-01-01"))
                .dt.total_seconds()
                .astype(float)
            )

        dtn = DateTime2Num(cols=self.deduced_types['datetime64'])
        self.steps.append(('Date2Num', dtn))
        self.steps.append(('Impute', self.imputer))
        trans = Pipeline(self.steps)

        # Keep the numeric matrix consistent with the original API while avoiding the deprecated category_encoders path.
        encoded_frame = pandas.DataFrame(
            trans.fit_transform(working_df),
            columns=working_df.columns,
        )
        self.transformed_df = encoded_frame.copy()

        rounded_columns = set(ordinal_columns).union(self.deduced_types['categorical'])
        for clmn in self.transformed_df.columns.intersection(rounded_columns):
            rounded = pandas.to_numeric(self.transformed_df[clmn], errors='coerce').round(0)
            if self.force_impute:
                self.transformed_df[clmn] = rounded.astype(int)
            else:
                self.transformed_df[clmn] = rounded.astype(float)

        for clmn in self.deduced_types['text']:
            txt_array = txt_clmn_df[clmn].values
            processed_txt = self.txt_prcsr.fit_transform(txt_array)
            length = processed_txt.shape[1]
            clmn_names = ["%s_%d" % (clmn, _) for _ in range(length)]
            self.transformed_df = pandas.concat(
                [self.transformed_df, pandas.DataFrame(processed_txt.toarray(), columns=clmn_names)], axis=1)

        self.encoded = True
        return self.transformed_df
