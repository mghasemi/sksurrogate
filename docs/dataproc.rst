======================================
Automated Data Type Recognition
======================================
In most ML scenarios, most of the development and deployment tasks deal
with data input pipelines. A data pipeline handles data intake, linkage,
type detection, and missing data imputation. SKSurrogate covers the later
two stapes automatically and allows for customization as well.
This is done via the ``DataProcess`` module.

For model selection and cross-validation, prefer
``FoldLocalDataPreprocessor`` inside an sklearn ``Pipeline``. Unlike the eager
``DataPreprocess.encode()`` workflow below, it learns imputations, categorical
levels, and text vocabularies during each pipeline fit. This keeps validation
folds out of preprocessing fits::

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import Pipeline
    from SKSurrogate import FoldLocalDataPreprocessor

    pipeline = Pipeline([
        ("preprocess", FoldLocalDataPreprocessor(scale_numeric=True)),
        ("model", LogisticRegression(max_iter=500)),
    ])
    scores = cross_val_score(pipeline, X_dataframe, y, cv=5)

The fold-local transformer uses median imputation for numeric and datetime
columns, one-hot encoding for boolean and categorical columns, and per-column
TF-IDF for pandas ``StringDtype`` columns. Scaling is optional and applies to
numeric and datetime features. Object/string columns are treated as
categorical; cast free-form text columns to ``pd.StringDtype()`` to select the
TF-IDF path. The transformer's input must remain a DataFrame with the same
columns at fit and prediction time. ``DataPreprocess`` remains available for
eager, whole-frame transformations and should not be run before a held-out
split when its fitted state could see validation or test rows.

Bundles trained from a preset are array-tolerant
------------------------------------------------

An ``AML`` preset search only installs ``FoldLocalDataPreprocessor`` when it is
given a ``DataFrame`` — that is what makes a mixed-type partition searchable
without leaking encodings across folds. A pipeline that starts with it then
accepts *only* frames, but several consumers legitimately hold positional
arrays: partial dependence and accumulated local effects vary one column of a
value matrix, and learning-curve helpers predict from ``X.to_numpy()``.

``frame_tolerant(estimator, columns=None)`` (in ``SKSurrogate.DataProcess``)
therefore prefixes such a pipeline with ``ArrayAsFrame``, which labels an array
with the training column names in order and passes a DataFrame through
untouched. The API applies it in ``_new_bundle``, so every bundle it writes
keeps both contracts:

.. code:: python

    from SKSurrogate import load_bundle

    bundle = load_bundle(path)
    bundle.model.predict(frame)          # serving path: a frame
    bundle.model.predict(frame.to_numpy())  # explanations: positional rows

    [name for name, _ in bundle.model.steps]
    # ['as_frame', 'preprocess', 'stp_0']

Caller-supplied search spaces and ``cat_cols``
----------------------------------------------

A preset search encodes mixed-type data by itself. A search space supplied by
the caller does not: ``AML`` only installs an encoder when it is told which
columns are categorical, so a config-based run on a dataset with a text column
used to fail with ``could not convert string to float``. ``fit_experiment_bundle``
and ``optimize_pipeline`` therefore detect the non-numeric feature columns and
pass them as ``cat_cols``; that encoder is a pipeline step, so it is still fitted
inside each cross-validation fold. The choice is recorded in the bundle's
reproducibility metadata:

.. code:: python

    bundle.metadata["reproducibility"]["configuration"]["cat_cols"]
    # ['surname']
    [name for name, _ in bundle.model.steps]
    # ['categorical_encoder', 'stp_0']

Note that a free-text identifier (a surname, a customer id) becomes one dummy
column per distinct value. Drop such columns from a search rather than encoding
them, and prefer the ``StringDtype`` TF-IDF path for genuinely free text.

Currently, the DataProcess module identifies the following data types
automatically:

  + Binary
  + Categorical
  + Date/Time
  + Float
  + Integer
  + Label
  + Text
  + Objects

We note that the Object type could include various types which may have a
known structure but are not implemented in the module yet.

The following example demonstrates the basic functions of the DataProcess module.

Example:
-----------------
Randomly generated dataframe with various types of data::

    import numpy as np
    import pandas as pd
    import random
    from lorem_text import lorem

    N = 100
    categorical = np.array([random.choice(['Cat01', 'Cat02', 'Cat03', 'Cate04', None]) for _ in range(N)])
    binary = np.array([random.choice(['Bin0', 'Bin1', None]) for _ in range(N)])
    float1 = np.random.uniform(low=-2.0, high=4.0, size=N)
    float2 = np.random.uniform(low=0.0, high=10.0, size=N)
    int1 = np.random.randint(0, high=20, size=N)

    def random_str(max_len=15):
        chars = [' '] + [chr(_) for _ in range(ord('a'), ord('z')+1)] + [' ']+  [chr(_) for _ in range(ord('A'), ord('Z')+1)] + [' ']
        ln = random.randint(0, max_len)
        rand_list = [random.choice(chars) for _ in range(ln)]
        return ''.join(rand_list)

    def random_date(init_date, date_range=30):
        offset = random.randint(0, date_range)
        new_date = np.datetime64(init_date) + offset
        return new_date

    strs = [random_str() for _ in range(N)]
    dates = [random_date("2021-03-01", 60) for _ in range(N)]

    texts = [lorem.sentence() for _ in range(N)]

    frame = dict()
    frame['categorical'] = categorical
    frame['binary'] = binary
    frame['float1'] = float1
    frame['float2'] = float2
    frame['int1'] = int1
    frame['str'] = strs
    frame['date'] = dates
    frame['txt'] = texts
    df = pd.DataFrame(frame)
    df = df.astype({'txt':pd.StringDtype()})

Import and process the sample dataframe::

    from SKSurrogate import *
    A = DataPreprocess(df)
    A.deduce_types()
    A.deduced_types

which returns::

    {'float64': ['float1', 'float2'],
     'int64': ['int1'],
     'datetime64': ['date'],
     'other': [],
     'text': ['txt'],
     'binary': ['binary'],
     'categorical': ['categorical'],
     'label': ['str'],
     'obsolete': []}

Then::

    A.encode()
    print(A.steps)

The output is a SK-Learn compatible pipeline::

    [('OneHot',
      OneHotEncoder(cols=['categorical'], drop_invariant=True,
                    handle_missing='return_nan', handle_unknown='return_nan')),
     ('Ordinal',
      OrdinalEncoder(cols=['binary', 'str'], handle_missing='return_nan',
                     handle_unknown='return_nan',
                     mapping=[{'col': 'binary', 'mapping': {'Bin0': 0, 'Bin1': 1}},
                              {'col': 'str',
                               'mapping': {'': 0, ' ': 1, ' DmTHDRQErIhF': 2,
                                           ' FmseqO': 3, ' j knr': 4, ' pcG': 5,
                                           'AVJVq nsqyHRpM': 6, 'AYihJxhUbN ': 7,
                                           'Agpg': 8, 'C': 9, 'CKJ': 10, 'CcnGK': 11,
                                           'D': 12, 'DkMstNYdjoRj ': 13, 'EITp': 14,
                                           'FAWrCVv': 15, 'FKgFwuGLmQqLR': 16,
                                           'FVtvoWBCEEi': 17, 'G T oVPh': 18,
                                           'GAxyFGqpzrJXe': 19, 'GYfNntcQww': 20,
                                           'HB euaV YFIb': 21, 'IENmSCFiAECp': 22,
                                           'IGZRolGBCKLsyg': 23,
                                           'J mNPFImkjd iRw': 24, 'JW': 25,
                                           'KSKpIlRm': 26, 'KevYeZyrsvwY': 27,
                                           'KhNjalpZkqxFGBC': 28,
                                           'KjtCfjg PZrx k ': 29, ...}}])),
     ('Date2Num', DateTime2Num(cols=['date'])),
     ('Impute', IterativeImputer())]
