======================================
Automated Data Type Recognition
======================================
In most ML scenarios, most of the development and deployment tasks deal
with data input pipelines. A data pipeline handles data intake, linkage,
type detection, and missing data imputation. SKSurrogate covers the later
two stapes automatically and allows for customization as well.
This is done via the ``DataProcess`` module.

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
