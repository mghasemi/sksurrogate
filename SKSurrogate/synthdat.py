"""
***********************************
synthdata Module
***********************************
This module provides basic framework for generating synthetic data resembling an existing dataset.
One could determine types of each field and the possible values for each field. Then the `InspectData` class will
produce data based on the given types. Moreover, one can associate marginal distributions to each field or a joint
distribution for data generation. If no distribution is associated, then the data will be generated uniformly over
the required ranges.

Supported data types
***********************************
The following data types are supporeted:

    + :`SynthBin`: Support for binary data, i.e., the data fields consisting of 0, 1 values;
    + :`SynthInt`: Support for integer valued data;
    + :`SynthReal`: Support for real valued data;
    + :`SynthCat`: Support for categorical type of data, i.e., the discrete variables whose values are predetermined;
    + :`SynthDate`: Support for datetime data;

Each of data types accept `data` that is a 1d `numpy.array` and `rv` which is a `scipy.stats` distribution implementing
`rv.rvs` to generate samples.
Among the above `SynthInt`, `SynthReal` accept two parameters `a` and `b` which are the lower and upper bounds of the
sampling interval respectively. `SynthDate` accepts `frmt` which determines the date formatting for the input data.

Generating Synthetic Data
***********************************
The `SynthData` class is responsible for generating synthetic data based on types, distributions and relations
defined on the data. One initiates an instant as::

    sd = SynthData(df, default_rv='uniform', distribution_type='marginal, rv=None)

where `df` is the pandas dataframe that will be simulated. The rest of arguments are optional:

    + `default_rv` determines the default distribution for those fields where no distribution is associated to. If the `distribution_type` is set to 'joint' this will be ignored.
    + `distribution_type` determines if the distribution(s) calculated based on `df` are *marginals* or a single *joint* distribution.
    + `rv` determines a predefined distribution for joint distribution.

To set the type of a column of `df` one should use `set_type` method. This method accepts a list of columns name, their
type and a tuple of initiating parameters. Every column in the column's list will be given the same type. The type could
be either an instance of `SynthBin`, `SynthInt`, `SynthReal`, `SynthCat`, `SynthDate`, or a string determining the type,
e.g., 'bin', 'int', 'real', 'cat', 'date'. If no type is associated to a column, it is assumed to be of categorical type.

If it is required to impose a constraint on a column, on can use `where()` method. The statement to add a constraint
would look like `sd.where(field('clmn1') > val1)` or `sd.where(field(clmn1) <= field(clmn2))`. The acceptable operators
are `==`, `!=`, `>`, `<`, `>=`, `<=`, `in`, `nin`. The operators `in` and `nin` check membership of elements of 'clmn1'
in 'val' which has to be an iterable or membership in the column 'clmn2', clearly `in` stands for belonging and `nin`
stants for *not in*.

The final command which generates the synthetic data is `sample(num)` where 'num' is the number of synthetic samples to
be generated. This method will return a `pandas.DataFrame` containing synthetic data of size 'num'.

Constraints
***********************************
It is quite common that the values of some fields in a record depend on other fields. Simple constraints on the
values of a field and relations between pairs can be handled using `field` objects.
"""

from datetime import datetime, timedelta
from numpy import inf, array, unique, cov, transpose
from pandas import DataFrame, concat


## Synthetic data types
class SynthBase(object):
    """
    The base class for various synthetic data types.
    """

    def __init__(self, data=None, rv=None):
        self.data = data
        self.rv = rv

    def get_val(self, x):
        """
        Coverts the value of `x` into numeric that can be handled by random distributions
        :param x: the value to be converted into numeric
        :return: the corresponding numeric value
        """
        return x

    def set_val(self):
        """
        Generates the numeric translation of the given data.
        :return:
        """
        return [self.get_val(x) for x in self.data]

    def ret_val(self, X):
        """
        Similar to `set_val` but works on the data stored in `X`
        :param X: the data to be converted to numbers
        :return: translation of `X`
        """
        return [self.get_val(x) for x in X]


class SynthBin(SynthBase):
    """
    Support for binary data, i.e., the data fields consisting of 0, 1 values
    """

    def __init__(self, data=None, rv=None):
        super(SynthBin, self).__init__(data=data, rv=rv)

    def get_val(self, x):
        if x < 0.5:
            return 0
        else:
            return 1

    def set_val(self):
        lst = [self.get_val(x) for x in self.data]
        return lst

    def ret_val(self, X):
        return X


class SynthInt(SynthBase):
    """
    Support for integer valued data;
    """

    def __init__(self, a=None, b=None, data=None, rv=None):
        super(SynthInt, self).__init__(data=data, rv=rv)
        m = min(data) if data is not None else -inf
        M = max(data) if data is not None else inf
        self.m = round(a) if a is not None else m
        self.M = round(b) if b is not None else M

    def get_val(self, x):
        cndd = int(max(min(self.M, round(x)), self.m))
        return cndd

    def set_val(self):
        lst = [self.get_val(x) for x in self.data]
        return lst

    def ret_val(self, X):
        return [self.get_val(x) for x in X]


class SynthReal(SynthBase):
    """
    Support for real valued data;
    """

    def __init__(self, a=None, b=None, data=None, rv=None):
        super(SynthReal, self).__init__(data=data, rv=rv)
        m = min(data) if data is not None else -inf
        M = max(data) if data is not None else inf
        self.m = a if a is not None else m
        self.M = b if b is not None else M

    def get_val(self, x):
        cndd = max(min(self.M, x), self.m)
        return cndd

    def set_val(self):
        lst = [self.get_val(x) for x in self.data]
        return lst

    def ret_val(self, X):
        return X


class SynthCat(SynthBase):
    """
    Support for categorical type of data, i.e., the discrete variables whose values are predetermined;
    """

    def __init__(self, data, rv=None):
        super(SynthCat, self).__init__(data=data, rv=rv)
        itms = self.get_uniques(array(data))
        self.items = list(itms)
        self.m = 0
        self.M = len(self.items) - 1

    def get_val(self, x):
        cndd = int(max(min(self.M, round(x)), self.m))
        return self.items[cndd]

    def set_val(self):
        lst = [self.items.index(x) for x in self.data]
        return lst

    def ret_val(self, X):
        return [self.get_val(x) for x in X]

    @staticmethod
    def get_uniques(self, X):
        lst = unique(X)
        return lst


class SynthDate(SynthBase):
    """
    Support for datetime data;
    """

    def __init__(self, data, frmt="%Y-%m-%d", rv=None):
        super(SynthDate, self).__init__(data=data, rv=rv)
        self.dates = [datetime.strptime(x, frmt) for x in data]
        self.m_date = min(self.dates)
        self.M_date = max(self.dates)
        self.m = 0
        self.M = (self.M_date - self.m_date).days

    def set_val(self):
        dys = [(x - self.m_date).days for x in self.dates]
        return dys

    @staticmethod
    def get_val(self, x):
        cndd = max(min(self.M, round(x)), self.m)
        cndd_date = self.m_date + timedelta(days=cndd)
        return cndd_date

    def ret_val(self, X):
        return [self.get_val(x) for x in X]


## Constraints in data generation
class field(object):
    """
    A generic class to handel simple constraints on columns.
    Accepts only one parameter which refers to a column in the DataFrame.
    """

    def __init__(self, fld):
        self.fld = fld
        self.op = None
        self.other = None

    def __eq__(self, val):
        self.op = "=="
        self.other = val
        return self

    def __ne__(self, val):
        self.op = "!="
        self.other = val
        return self

    def __gt__(self, val):
        self.op = ">"
        self.other = val
        return self

    def __lt__(self, val):
        self.op = "<"
        self.other = val
        return self

    def __ge__(self, val):
        self.op = ">="
        self.other = val
        return self

    def __le__(self, val):
        self.op = "<="
        self.other = val
        return self

    def isin(self, val):
        self.op = "in"
        self.other = val
        return self

    def notin(self, val):
        self.op = "nin"
        self.other = val
        return self


## Synthetic data generator
class SynthData(object):
    """
    A class which takes a *real* `pandas.DataFrame` and produces *synthetic* data similar to the real data based on
    types and distributions provided by the user and/or extracted out of original data.
    :param df: a `pandas.DataFrame` containing original data.
    :param default_rv: default distribution of columns; default 'uniform'. Also could be 'normal'. Only effective if `distribution_type` is 'marginal', otherwise will be ignored.
    :param distribution_type: default 'marginal'. Determines the type of distribution. If 'joint', then either a normal distribution is calculated based on provided data or will use `rv` if `rv` is not 'None'.
    :param rv: default 'None'. The joint distribution of variables. Only effective if `distribution_type` is 'joint'.
    """

    def __init__(self, df, default_rv="uniform", distribution_type="marginal", rv=None):
        self.df = df
        self.default_rv = default_rv.lower()
        self.distribution_type = distribution_type.lower()
        self.rv = rv
        self.transformed = {}
        self.tr_df = None
        self.columns = df.columns
        self.types = {}
        self.params = {}
        self.synth = {}
        self.SynthObj = {}
        self.const = []

    def set_type(self, clmns, typ, param=None):
        """
        Define the type of columns.
        :param clmns: a *list* of 'df' columns
        :param typ: the associated type, either an string ('bin', 'int', 'real', 'cat', 'date) or an instance of `SynthBin`, `SynthInt`, `SynthReal`, `SynthCat`, `SynthDate`.
        :param param: parameters to be passed to synthetic data type if an string is given for 'typ'. It could be a couple (a, b) for 'int' and 'real' type and just the format for 'date'.
        """
        for clmn in clmns:
            self.types[clmn] = typ
            self.params[clmn] = param

    def transform(self):
        """
        *internal* to analyse and initialize data types and convert them to numerical values.
        :return: `None`
        """
        for clm in self.columns:
            data = list(self.df[clm])
            a, b = None, None
            frmt = "%Y-%m-%d"
            if clm in self.types:
                if not isinstance(
                    self.types[clm],
                    (SynthBin, SynthInt, SynthReal, SynthCat, SynthDate),
                ):
                    typ = self.types[clm].lower()
                    if typ == "bin":
                        SynthObj = SynthBin(data)
                    elif typ == "int":
                        if self.params[clm] is not None:
                            a = self.params[clm][0]
                            b = self.params[clm][1]
                        SynthObj = SynthInt(a=a, b=b, data=data)
                    elif typ == "real":
                        if self.params[clm] is not None:
                            a = self.params[clm][0]
                            b = self.params[clm][1]
                        SynthObj = SynthReal(a=a, b=b, data=data)
                    elif typ == "cat":
                        SynthObj = SynthCat(data)
                    elif typ == "date":
                        if self.params[clm] is not None:
                            frmt = self.params[clm]
                        SynthObj = SynthDate(data, frmt=frmt)
                else:
                    SynthObj = self.types[clm]
            else:
                SynthObj = SynthCat(data)
            lst = SynthObj.set_val()
            self.SynthObj[clm] = SynthObj
            self.transformed[clm] = lst
            self.tr_df = DataFrame(self.transformed)

    def generate(self, num):
        """
        *internal* Generate 'num' synthetic data records without considering constraints
        :param num: number of samples
        :return: `pandas.DataFrame`
        """
        from scipy.stats import norm
        from scipy.stats import uniform

        v = self.tr_df[self.columns].values
        self.mean = v.mean(axis=0)
        self.var = v.var(axis=0)
        self.m = v.min(axis=0)
        self.M = v.max(axis=0)
        self.synth = {}
        idx = 0
        # if the default distribution is set to `marginal`
        if self.distribution_type == "marginal":
            self.rv = [None for i in range(self.m.shape[0])]
            for clm in self.columns:
                if self.SynthObj[clm].rv is not None:
                    self.rv[idx] = self.SynthObj[clm].rv
                elif self.default_rv == "uniform":
                    self.rv[idx] = uniform(
                        loc=self.m[idx], scale=self.M[idx] - self.m[idx]
                    )
                else:
                    self.rv[idx] = norm(loc=self.mean[idx], scale=self.var[idx])
                data = list(self.rv[idx].rvs(num))
                lst = self.SynthObj[clm].ret_val(data)
                self.synth[clm] = lst
                self.synth_df = DataFrame(self.synth)
                idx += 1
        # if the default distribution type is set to `joint`
        else:
            if self.rv is None:
                from scipy.stats import multivariate_normal

                self.cov = cov(transpose(v))
                self.rv = multivariate_normal(self.mean, self.cov, allow_singular=True)
            X = self.rv.rvs(num)
            idx = 0
            for clm in self.columns:
                data = list(X[:, idx])
                lst = self.SynthObj[clm].ret_val(data)
                self.synth[clm] = lst
                self.synth_df = DataFrame(self.synth)
                idx += 1
        return self.synth_df

    def where(self, cns):
        """
        Add a constraint of values of a column using `field` objects.
        :param cns: the constraint like `field(clmn1) > val1` or `field(clmn1) <= field(clmn2)`.
        :return: `None`
        """
        self.const.append(cns)

    def filter(self, df):
        """
        Filter the 'df' to remove illegal records according to constraints
        :param df: the dataframe to be filtered
        :return: the filtered dataframe
        """
        t_df = df
        for cnd in self.const:
            if isinstance(cnd.other, field):
                if cnd.op == "==":
                    t_df = t_df[t_df[cnd.fld] == t_df[cnd.other.fld]]
                elif cnd.op == "!=":
                    t_df = t_df[t_df[cnd.fld] != t_df[cnd.other.fld]]
                elif cnd.op == ">":
                    t_df = t_df[t_df[cnd.fld] > t_df[cnd.other.fld]]
                elif cnd.op == "<":
                    t_df = t_df[t_df[cnd.fld] < t_df[cnd.other.fld]]
                elif cnd.op == ">=":
                    t_df = t_df[t_df[cnd.fld] >= t_df[cnd.other.fld]]
                elif cnd.op == "<=":
                    t_df = t_df[t_df[cnd.fld] <= t_df[cnd.other.fld]]
                elif cnd.op == "in":
                    t_df = t_df[t_df[cnd.fld].isin(t_df[cnd.other.fld])]
                elif cnd.op == "nin":
                    t_df = t_df[~t_df[cnd.fld].isin(t_df[cnd.other.fld])]
            else:
                if cnd.op == "==":
                    t_df = t_df[t_df[cnd.fld] == cnd.other]
                elif cnd.op == "!=":
                    t_df = t_df[t_df[cnd.fld] != cnd.other]
                elif cnd.op == ">":
                    t_df = t_df[t_df[cnd.fld] > cnd.other]
                elif cnd.op == "<":
                    t_df = t_df[t_df[cnd.fld] < cnd.other]
                elif cnd.op == ">=":
                    t_df = t_df[t_df[cnd.fld] >= cnd.other]
                elif cnd.op == "<=":
                    t_df = t_df[t_df[cnd.fld] <= cnd.other]
                elif cnd.op == "in":
                    t_df = t_df[t_df[cnd.fld].isin(cnd.other)]
                elif cnd.op == "nin":
                    t_df = t_df[~t_df[cnd.fld].isin(cnd.other)]
        return t_df

    def sample(self, num):
        """
        Produces 'num' records of synthetic data following given types, distributions and constraints
        :param num: number of synthetic data records
        :return: a dataframe consisting of 'num' synthetic records.
        """
        res_df = DataFrame()
        self.transform()
        g_num = 0
        while g_num < num:
            f_df = self.generate(max(num - g_num, 2))
            f_df = self.filter(f_df)
            res_df = concat([res_df, f_df], ignore_index=True)
            g_num = len(res_df)
        return res_df
