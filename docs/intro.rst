=============================
Introduction
=============================
``SKSurrogate`` is a suite of tools focused on designing and tuning machine learning pipelines and
track the evolution of simple modeling tasks. ``SKSurrogate`` is designed to employ Surrogate
Optimization technique in a flexible way which enjoys from the wealth of existing well-known python
tools. It provides a ui to perform surrogate optimization on general functions and has a familiar
ui to perform hyperparameter tuning for scikit-learn compatible models.
Moreover, ``SKSurrogate`` uses surrogate optimization and evolutionary optimization algorithms to
construct and tune complex pipelines (automl) based on a set of given (scikit-learn compatible) models, like
`TPOT <https://epistasislab.github.io/tpot/>`_ does.

Note that automl part is not designed to result in outstanding results in a few minutes, rather it
needs hours if not days to come up with a good result. There are various methods that speeds up the
pipeline design/optimization process such as selecting faster surrogates, fewer number of steps and
evolutionary optimization settings.

The evolutionary optimization module is designed to be very flexible and can be modified to perform
evolutionary optimization on any given evolutionary compatible problem.

Dependencies
=============================

    - `NumPy <http://www.numpy.org/>`_,
    - `scipy <https://www.scipy.org/>`_,
    - `pandas <https://pandas.pydata.org/>`_,
    - `matplotlib <https://matplotlib.org/>`_,
    - `scikit-learn <https://scikit-learn.org/stable/>`_,
    - `ELI5 <https://github.com/TeamHG-Memex/eli5>`_,
    - `SALib <https://github.com/SALib/SALib>`_,
    - `peewee <https://github.com/coleifer/peewee>`_.

Download
=============================
`SKSurrogate` can be obtained from `https://github.com/mghasemi/sksurrogate <https://github.com/mghasemi/sksurrogate>`_.

Installation
=============================
To install `SKSurrogate`, run the following in terminal::

    sudo python setup.py install

Documentation
=============================
The documentation is produced by `Sphinx <http://www.sphinx-doc.org/en/stable/>`_ and is intended to cover code usage
as well as a bit of theory to explain each method briefly.
For more details refer to the documentation at `sksurrogate.rtfd.io <http://sksurrogate.readthedocs.io/>`_.

License
=============================
This code is distributed under `MIT license <https://en.wikipedia.org/wiki/MIT_License>`_:

MIT License
-----------------------------

    Copyright (c) 2022 Mehdi Ghasemi

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.