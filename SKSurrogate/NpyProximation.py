"""
Hilbert Space based regression
==================================
"""

Infinitesimal = 1e-7


class Error(Exception):
    r"""
    Generic errors that may occur in the course of a run.
    """

    def __init__(self, *args):
        super(Error, self).__init__(*args)


class Measure(object):
    r"""
    Constructs a measure :math:`\mu` based on `density` and `domain`.

    :param density: the density over the domain:
            + if none is given, it assumes uniform distribution

            + if a callable `h` is given, then :math:`d\mu=h(x)dx`

            + if a dictionary is given, then :math:`\mu=\sum w_x\delta_x` a discrete measure.
              The points :math:`x` are the keys of the dictionary (tuples) and the weights :math:`w_x` are the values.
    :param domain: if `density` is a dictionary, it will be set by its keys. If callable, then `domain` must be a list
                   of tuples defining the domain's box. If None is given, it will be set to :math:`[-1, 1]^n`
    """

    def __init__(self, density=None, domain=None, **kwargs):
        from types import FunctionType
        # set the density
        if density is None:
            self.density = lambda x: 1.
        elif type(density) is FunctionType:
            self.density = density
        elif type(density) is dict:
            self.density = density  # lambda x: density[x] if x in density else 0.
        else:
            raise Error("The `density` must be either a callable or a dictionary of real numbers.")
        # check and set the domain
        self.continuous = True
        self.dim = 0
        if type(domain) is list:
            self.dim = len(domain)
            for intrvl in domain:
                if (type(intrvl) not in [list, tuple]) or (len(intrvl) != 2):
                    raise Error("`domain` should be a list of 2-tuples.")
            self.supp = domain
        elif type(density) is dict:
            self.supp = density.keys()
            self.continuous = False
        else:
            raise Error("No domain is specified.")

    def integral(self, f):
        r"""
        Calculates :math:`\int_{domain} fd\mu`.

        :param f: the integrand
        :return: the value of the integral
        """
        from types import FunctionType
        m = 0.
        if type(f) not in [dict, FunctionType]:
            raise Error("The integrand must be a `function` or a `dict`")
        if type(f) == dict:
            fn = lambda x: f[x] if x in f else 0.
        else:
            fn = f
        if self.continuous:
            from scipy import integrate
            fw = lambda *x: fn(*x) * self.density(*x)
            m = integrate.nquad(fw, self.supp)[0]
        else:
            for p in self.supp:
                # print(p, self.density[p], fn)
                # print(p, self.density[p], fn(p))
                m += self.density[p] * fn(p)
        return m

    def norm(self, p, f):
        r"""
        Computes the norm-`p` of the `f` with respect to the current measure,
        i.e., :math:`(\int_{domain}|f|^p d\mu)^{1/p}`.

        :param p: a positive real number
        :param f: the function whose norm is desired.
        :return: :math:`\|f\|_{p, \mu}`
        """
        from math import pow
        absfp = lambda *x: pow(abs(f(*x)), p)
        return pow(self.integral(absfp), 1. / p)


class FunctionBasis(object):
    """
    This class generates two typical basis of functions: Polynomials and Trigonometric
    """

    def __init__(self):
        pass

    def Poly(self, n, deg):
        """
        Returns a basis consisting of polynomials in `n` variables of degree at most `deg`.

        :param n: number of variables
        :param deg: highest degree of polynomials in the basis
        :return: the raw basis consists of polynomials of degrees up to `n`
        """
        from itertools import product
        from numpy import prod
        B = []
        for o in product(range(deg + 1), repeat=n):
            if sum(o) <= deg:
                B.append(lambda x, e=o: prod([x[i] ** e[i] for i in range(n)]))
        return B

    def Fourier(self, n, deg, l=1.):
        """
        Returns the Fourier basis of degree `deg` in `n` variables with period `l`

        :param n: number of variables
        :param deg: the maximum degree of trigonometric combinations in the basis
        :param l: the period
        :return: the raw basis consists of trigonometric functions of degrees up to `n`
        """

        from numpy import sin, cos, prod
        from itertools import product
        B = [lambda x: 1.]
        E = list(product([0, 1], repeat=n))
        RawCoefs = list(product(range(deg + 1), repeat=n))
        Coefs = set()
        for prt in RawCoefs:
            p_ = list(prt)
            p_.sort()
            Coefs.add(tuple(p_))
        for o in Coefs:
            if (sum(o) <= deg) and (sum(o) > 0):
                for ex in E:
                    if sum(ex) > 0:
                        f_ = lambda x, o=o, ex=ex: prod(
                            [sin(o[i] * x[i] / l) ** ex[i] * cos(o[i] * x[i] / l) ** (1 - ex[i]) if o[i] > 0 else 1. for
                             i in range(n)])
                        B.append(f_)
        return B


class FunctionSpace(object):
    r"""
    A class tha facilitates a few types of computations over function spaces of type :math:`L_2(X, \mu)`

    :param dim: the dimension of 'X' (default: 1)
    :param measure: an object of type `Measure` representing :math:`\mu`
    :param basis: a finite basis of functions to construct a subset of :math:`L_2(X, \mu)`
    """
    dim = 1  # type: int

    def __init__(self, dim=1, measure=None, basis=None):
        self.dim = int(dim)
        if (measure is not None) and (isinstance(measure, Measure)):
            self.measure = measure
        else:
            # default measure is set to be the Lebesgue measure on [0, 1]^dim
            D = [(0., 1.) for _ in range(self.dim)]
            self.measure = Measure(domain=D)
        if basis is None:
            # default basis is linear
            from numpy import array
            B = [lambda x: 1.]
            for i in range(self.dim):
                B.append(lambda x, i=i: x[i] if type(x) is array else x)
            self.base = B
        else:
            self.base = basis
        self.OrthBase = []
        self.Gram = None

    def inner(self, f, g):
        r"""
        Computes the inner product of the two parameters with respect to
        the measure `measure`, i.e., :math:`\int_Xf\cdot g d\mu`.

        :param f: callable
        :param g: callable
        :return: the quantity of :math:`\int_Xf\cdot g d\mu`
        """
        fn = lambda x, f_=f, g_=g: f_(x) * g_(x)
        return self.measure.integral(fn)

    def project(self, f, g):
        r"""
        Finds the projection of `f` on `g` with respect to the inner
        product induced by the measure `measure`.

        :param f: callable
        :param g: callable
        :return: the quantity of :math:`\frac{\langle f, g\rangle}{\|g\|_2}g`
        """
        a = self.inner(f, g)
        b = self.inner(g, g)
        return lambda x: a * g(x) / b

    def GramMat(self):
        from numpy import array
        N = len(self.base)
        cfs = array([[0.] * N] * N)
        for i in range(N):
            for j in range(i, N):
                cf = self.inner(self.base[i], self.base[j])
                cfs[i][j] = cf
                cfs[j][i] = cf
        self.Gram = cfs

    def minor_gram(self, i):
        from numpy import array
        if self.Gram is None:
            self.GramMat()
        return array([[self.Gram[idx][jdx] for idx in range(i + 1)] for jdx in range(i + 1)])

    def minor(self, i, j):
        from numpy import array, delete
        from numpy.linalg import det
        if j == 1:
            return 1.
        cfs = array([[0.] * j] * (j - 1))
        for jdx in range(j):
            for idx in range(j - 1):
                cfs[idx][jdx] = self.Gram[idx][jdx]
        return det(delete(cfs, i, 1))

    def FormBasis(self):
        """
        Call this method to generate the orthogonal basis corresponding
        to the given basis.
        The result will be stored in a property called ``OrthBase`` which
        is a list of function that are orthogonal to each other with
        respect to the measure ``measure`` over the given range ``domain``.
        """
        from numpy.linalg import det
        from numpy import sqrt
        N = len(self.base)
        GramDets = [1.] + [det(self.minor_gram(i)) for i in range(N)]
        B = []
        D = []
        for j in range(1, N + 1):
            j_ = j
            cf = [(-1) ** (i + j - 1) * self.minor(i, j_) / sqrt(GramDets[j_ - 1] * GramDets[j_]) for i in range(j_)]
            # print(j_, cf, [self.base[i](2.) for i in range(j_)])
            B.append(lambda x: sum([cf[i] * self.base[i](x) for i in range(j_)]))
            D.append(cf)
        self.OrthBase = []
        for i in range(len(D)):
            fn = lambda x, i=i: sum([D[i][j] * self.base[j](x) for j in range(len(D[i]))])
            self.OrthBase.append(fn)

    def Series(self, f):
        r"""
        Given a function `f`, this method finds and returns the
        coefficients of the	series that approximates `f` as a
        linear combination of the elements of the orthogonal basis :math:`B`.
        In symbols :math:`\sum_{b\in B}\langle f, b\rangle b`.

        :return: the list of coefficients :math:`\langle f, b\rangle` for :math:`b\in B`
        """
        cfs = []
        for b in self.OrthBase:
            cfs.append(self.inner(f, b))
        return cfs


class Regression(object):
    """
    Given a set of points, i.e., a list of tuples of the equal lengths `P`, this class computes the best approximation
    of a function that fits the data, in the following sense:

        + if no extra parameters is provided, meaning that an object is initiated like ``R = Regression(P)`` then
          calling ``R.fit()`` returns the linear regression that fits the data.
        + if at initiation the parameter `deg=n` is set, then ``R.fit()`` returns the polynomial regression of
          degree `n`.
        + if a basis of functions provided by means of an `OrthSystem` object (``R.SetOrthSys(orth)``) then
          calling ``R.fit()`` returns the best approximation that can be found using the basic functions of the `orth` object.

    :param point: a list of points to be fitted or a callable to be approximated
    :param dim: dimension of the domain
    """

    def __init__(self, points, dim=None):
        from numpy import array, ndarray
        self.Points = None
        if type(points) in [list, array, ndarray]:
            self.Points = list(points)
            self.dim = len(points[0]) - 1
            supp = {}
            for p in points:
                supp[tuple(p[:-1])] = 1.
            self.meas = Measure(supp)
            self.f = lambda x: sum([p[-1] * (1 * (abs(x - array(p[:-1])) < 1.e-4)).min() for p in points])
        elif callable(points):
            if dim is None:
                raise Error("The dimension can not be determined")
            else:
                self.dim = dim
            self.f = points
            self.meas = Measure(domain=[(-1., 1.) for _ in range(self.dim)])
        self.Orth = FunctionSpace(dim=self.dim, measure=self.meas)
        # self.Orth.FormBasis()

    def SetMeasure(self, meas):
        """
        Sets the default measure for approximation.

        :param meas: a measure.Measure object
        :return: None
        """
        assert (isinstance(meas, Measure)), "SetMeasure accepts a NpyProximation.Measure object."
        self.meas = meas

    def SetFuncSpc(self, sys):
        """
        Sets the bases of the orthogonal basis

        :param sys: `orthsys.OrthSystem` object.
        :return: None

        .. Note::
            For technical reasons, the measure needs to be given via `SetMeasure` method. Otherwise, the Lebesque
            measure on :math:`[-1, 1]^n` is assumed.
        """
        assert (self.dim == sys.dim), "Dimensions of points and the orthogonal system do not match."
        sys.measure = self.meas
        self.Orth = sys
        self.Orth.FormBasis()

    def fit(self):
        """
        Fits the best curve based on the optional provided orthogonal basis.
        If no basis is provided, it fits a polynomial of a given degree (at initiation)
        :return: The fit.
        """
        coefs = self.Orth.Series(self.f)
        aprx = lambda x: sum([coefs[i] * self.Orth.OrthBase[i](x) for i in range(len(self.Orth.OrthBase))])
        return aprx


from sklearn.base import BaseEstimator, RegressorMixin


class HilbertRegressor(BaseEstimator, RegressorMixin):
    r"""
    Regression using Hilbert Space techniques Scikit-Learn style.

    :param deg: int, default=3
        The degree of polynomial regression. Only used if `base` is `None`
    :param base: list, default = None
        a list of function to form an orthogonal function basis
    :param meas: NpyProximation.Measure, default = None
        the measure to form the :math:`L_2(\mu)` space. If `None` a discrete measure will be constructed based on `fit` inputs
    :param fspace: NpyProximation.FunctionBasis, default = None
        the function subspace of :math:`L_2(\mu)`, if `None` it will be initiated according to `self.meas`
    """

    def __init__(self, deg=3, base=None, meas=None, fspace=None):
        self.deg = deg
        self.meas = meas
        self.base = base
        self.fspace = fspace

    def fit(self, X, y):
        """

        :param X: Training data
        :param y: Target values
        :return: `self`
        """
        from numpy import concatenate
        if len(X.shape) != 2:
            X = X.reshape(X.shape[0], 1)
        points = concatenate((X, y.reshape(y.shape[0], 1)), axis=1)
        self.Regressor = Regression(points)
        self.dim = X[0].shape[0]
        if self.fspace is not None:
            self.Regressor.SetFuncSpc(self.fspace)
        else:
            bs = FunctionBasis()
            B = bs.Poly(n=self.dim, deg=self.deg) if self.base is None else self.base
            self.fspace = FunctionSpace(dim=self.dim, basis=B)
            self.Regressor.SetFuncSpc(self.fspace)
        if self.meas is not None:
            self.Regressor.SetMeasure(self.meas)
        self.apprx = self.Regressor.fit()

        return self

    def predict(self, X):
        """
        Predict using the Hilbert regression method

        :param X: Samples
        :return: Returns predicted values
        """
        from numpy import array
        if len(X.shape) != 2:
            X = X.reshape(X.shape[0], 1)
        return array([self.apprx(x) for x in X])
