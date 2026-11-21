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
    """Map a date range onto ``[lower, upper]`` (linear rescale).

    Accepts either datetime values or numeric seconds, because
    :meth:`DataPreprocess.encode` converts columns to epoch seconds before this
    step. The rescale is fit on ``min``/``max``, so:

    * a constant column maps to ``lower`` instead of dividing by a zero span
      (it used to emit ``inf``/``nan``);
    * the divisor uses the column's own unit (``unit`` for datetimes, seconds for
      numeric input) rather than always comparing against a ``timedelta64`` — the
      old code raised ``numpy``'s ``UFuncTypeError`` on numpy >= 2 for numeric
      input, which is exactly what ``encode()`` feeds it.

    Prefer :class:`DatetimeCentering` for model features: a min/max rescale depends
    on the observed extremes, so it shifts whenever the data is refitted, and a
    single outlier compresses every other value.

    :param cols: default: `None`; column names for transformation.
    :param unit: default: 'day`; the time unit.
    :param lower: default: `0.0`; the lower bound of the transformation range.
    :param upper: default: `2.0`; the upper bound of the transformation range.
    """

    _UNITS = {
        "second": numpy.timedelta64(1, "s"),
        "minute": numpy.timedelta64(1, "m"),
        "hour": numpy.timedelta64(1, "h"),
        "day": numpy.timedelta64(1, "D"),
    }

    def __init__(self, cols=None, unit='day', lower=0., upper=2.):
        if unit not in self._UNITS:
            raise ValueError("`unit` must be either 'day', 'hour', 'minute', or 'second'.")
        self.cols = cols
        self.unit = unit
        self.lower = lower
        self.upper = upper
        self.coefs = dict()

    def fit(self, X):
        """Fits and calculates the transformation parameters for date range transformation.

        :param X: Input data
        :return: `self`
        """
        if self.cols is None:
            return self
        for clmn in self.cols:
            values = X[clmn].dropna()
            if not len(values):
                self.coefs[clmn] = {"kind": "empty", "anchor": 0.0, "coef": 0.0}
                continue
            if pandas.api.types.is_datetime64_any_dtype(values):
                anchor = values.min()
                span = (values.max() - anchor) / self._UNITS[self.unit]
                kind = "datetime"
            else:
                numeric = pandas.to_numeric(values, errors="coerce").dropna()
                anchor = float(numeric.min()) if len(numeric) else 0.0
                span = float(numeric.max()) - anchor if len(numeric) else 0.0
                kind = "numeric"
            # A zero span (constant column) maps everything to `lower`.
            coef = 0.0 if not numpy.isfinite(span) or span == 0 else (self.upper - self.lower) / span
            self.coefs[clmn] = {"kind": kind, "anchor": anchor, "coef": float(coef)}
        return self

    def transform(self, X):
        """Apply the calculated parameters to transform the datetime columns.

        :param X: the input data
        :return: transformed data
        """
        if self.cols is None:
            return X
        TX = X
        for clmn in self.cols:
            params = self.coefs[clmn]
            if params["kind"] == "empty":
                TX[clmn] = numpy.nan
                continue
            if params["kind"] == "datetime":
                values = pandas.to_datetime(TX[clmn], errors="coerce")
                offset = (values - params["anchor"]) / self._UNITS[self.unit]
                TX[clmn] = self.lower + params["coef"] * offset.astype(float)
            else:
                values = pandas.to_numeric(TX[clmn], errors="coerce").astype(float)
                TX[clmn] = self.lower + params["coef"] * (values - params["anchor"])
        return TX

    def fit_transform(self, X, y=None, **fit_params):
        """Fits and transforms the input data.

        :param X: input data;
        :param y: target field; not required, but included to comply with scikit-learn's signature
        :param fit_params: included to comply with scikit-learn's signature
        :return: transformed DataFrame
        """
        self.fit(X)
        return self.transform(X)


def _datetime_frame_to_epoch_seconds(X):
    """Seconds since 1970-01-01 UTC for every column of ``X``.

    Kept as a module-level function because fitted pipelines pickle a reference to
    it; changing its behaviour would silently alter every existing bundle. New
    pipelines use :class:`DatetimeCentering`, which subtracts a fitted anchor first
    (raw epoch seconds make an intercept-plus-slope design ill-conditioned:
    ``cond ~ 1e11`` versus ``~1`` when centred).
    """
    transformed = pandas.DataFrame(index=X.index)
    for column in X.columns:
        values = pandas.to_datetime(X[column], errors="coerce")
        seconds = values.astype("int64").astype(float) / 1e9
        seconds[values.isna()] = numpy.nan
        transformed[column] = seconds
    return transformed


class DatetimeCentering(TransformerMixin, BaseEstimator):
    """Convert datetime columns to **seconds relative to a fitted per-column anchor**.

    Why not raw epoch seconds: a feature centred a billion seconds from the origin
    makes ``[1, x]`` ill-conditioned (``cond ~ 1e11``), inflates coefficient
    variance, and silently neutralises L2 penalties — a Ridge ``alpha`` applied to a
    ``1e-6``-scale slope barely regularises. Subtracting the anchor leaves the
    offset out of the design: the served feature is "seconds since the first
    training observation", which is what a trend term should mean.

    Why not the legacy ``[0, 2]`` rescale: min/max scaling is outlier-sensitive,
    changes with every refit (so folds and serving disagree), divides by zero for a
    constant column, and discards the unit.

    The subtraction is done in the column's own resolution, never through an int64
    nanosecond cast, so dates outside 1677..2262 (or a dateutil year-1 parse) yield
    a large but finite number instead of an ``OverflowError``. Unparseable values
    become ``NaN`` (imputed downstream), never ``inf``.
    """

    def __init__(self, columns=None):
        self.columns = columns

    @staticmethod
    def _parse(series):
        values = pandas.to_datetime(series, errors="coerce")
        tz = getattr(values.dtype, "tz", None)
        if tz is not None:
            # The anchor and the feature are both UTC; naive-local would make the
            # offset depend on the machine's timezone.
            values = values.dt.tz_convert("UTC").dt.tz_localize(None)
        return values

    def fit(self, X, y=None):
        if not isinstance(X, pandas.DataFrame):
            raise TypeError("DatetimeCentering requires a pandas DataFrame")
        columns = list(self.columns) if self.columns is not None else list(X.columns)
        self.columns_ = columns
        self.anchors_ = {}
        for column in columns:
            values = self._parse(X[column])
            self.anchors_[column] = None if values.isna().all() else values.min()
        self.n_features_in_ = len(columns)
        return self

    def transform(self, X):
        if not hasattr(self, "anchors_"):
            raise RuntimeError("DatetimeCentering has not been fitted")
        if not isinstance(X, pandas.DataFrame):
            raise TypeError("DatetimeCentering requires a pandas DataFrame")
        missing = [column for column in self.columns_ if column not in X.columns]
        if missing:
            raise ValueError("DatetimeCentering is missing column(s): %r" % missing)
        transformed = pandas.DataFrame(index=X.index)
        for column in self.columns_:
            values = self._parse(X[column])
            anchor = self.anchors_[column]
            if anchor is None:
                transformed[column] = numpy.nan
                continue
            # Timedelta division keeps the column's resolution and cannot overflow.
            transformed[column] = (values - anchor) / numpy.timedelta64(1, "s")
        return transformed.astype(float)

    def get_feature_names_out(self, input_features=None):
        return numpy.asarray(
            list(self.columns_) if self.columns_ is not None else input_features, dtype=object
        )


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
    datetimes are converted to seconds relative to a fitted anchor, imputed, and
    scaled when ``scale_numeric`` is set. Explicitly declared
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
                # Centred seconds, not raw epoch seconds: the offset carries no
                # information and makes the design ill-conditioned (see
                # DatetimeCentering).
                ("centering", DatetimeCentering()),
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
        frame = pandas.DataFrame(array, columns=pandas.Index(names))
        # A mixed-type matrix is an *object* array, so every rebuilt column would
        # start as ``object`` dtype. That is not cosmetic: a DataFrame-first
        # preprocessor decides per column whether it is numeric or categorical from
        # the dtype it sees, so fitting one on a positional array classified every
        # numeric column as categorical (measured: a 3-column frame expanded to 423
        # one-hot columns) and the fitted encoder then disagreed with the same
        # pipeline fitted on a real frame. ``infer_objects`` restores the numeric
        # dtypes (and leaves genuine text columns as ``object``), which is what makes
        # "fit on the frame" and "fit on ``frame.to_numpy()``" the same model — the
        # contract bagging and the learning-curve helper rely on.
        return frame.infer_objects()


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
    :class:`FoldLocalDataPreprocessor` — directly, behind the
    :class:`ArrayAsFrame` adapter, or as a ``category_encoders`` column-wise
    encoder whose ``cols`` are named columns. Nested cross-validation uses this
    to decide whether a candidate still needs an encoder in front of it: a
    bundle trained from a preset already carries one, and stacking a second
    encoder on top fed the pipeline an already-encoded matrix. The
    ``category_encoders`` check matters because ``AML``'s ``cat_cols`` path
    builds ``OneHotEncoder(cols=...)`` first, which requires a DataFrame input
    (named columns) — double-wrapping it behind ``FoldLocalDataPreprocessor``
    handed it a numpy matrix and it crashed with
    ``X does not contain the columns listed in cols``.

    An aggregating meta-estimator (voting, stacking, bagging, boosting, or a
    single-estimator wrapper such as
    :class:`~SKSurrogate.ensembling.SampleWeightRouter`) is a frame-preprocessing
    estimator exactly when every member it fits is one — otherwise nested
    cross-validation wraps the whole ensemble in a second encoder and each member's
    own encoder then receives an already-encoded matrix.
    """
    steps = getattr(estimator, "steps", None)
    if not steps:
        return _aggregated_preprocesses_frames(estimator)
    first_step = steps[0][1]
    if isinstance(first_step, ArrayAsFrame):
        steps = steps[1:]
        if not steps:
            return False
        first_step = steps[0][1]
    if isinstance(first_step, FoldLocalDataPreprocessor):
        return True
    # category_encoders encoders (``cat_cols`` path) work on named columns and
    # thus also expect a raw frame; treat them as already preprocessing.
    try:
        from category_encoders.utils import BaseEncoder as CategoryEncoder
    except Exception:  # pragma: no cover - optional dep absent
        return False
    return isinstance(first_step, CategoryEncoder)


def _declared_members(estimator):
    """The base estimators a meta-estimator declares, as a flat tuple (may be empty).

    Covers scikit-learn's aggregators — ``estimators`` as ``(name, estimator)``
    pairs on voting/stacking, a bare list on ``AdaBoost*`` — and its
    single-estimator wrappers (``estimator`` on bagging/boosting bases), so one
    predicate answers for a whole ensemble regardless of how it was assembled.
    """
    members = getattr(estimator, "estimators", None)
    if members is None:
        single = getattr(estimator, "estimator", None)
        if single is None:
            return ()
        return (single,)
    resolved = []
    for entry in members:
        if isinstance(entry, tuple) and len(entry) == 2:
            resolved.append(entry[1])
        else:
            resolved.append(entry)
    return tuple(resolved)


def _aggregated_preprocesses_frames(estimator):
    members = _declared_members(estimator)
    if not members:
        return False
    return all(preprocesses_frames(member) for member in members)


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
        population = len(self.values[clmn])
        if not population:
            # An all-missing column (or an empty frame) has no population to
            # divide by; it is not binary, and this used to raise
            # ZeroDivisionError from inside type deduction.
            return False
        ratio = len(self.uniques[clmn]) / population
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
        population = len(self.values[clmn])
        if not population:
            # Same population guard as is_bin: a column with no observed values
            # cannot be categorical, and the division would raise.
            return False
        ratio = len(self.uniques[clmn]) / population
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
