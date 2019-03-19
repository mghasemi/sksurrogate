try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

Description = "A python package that implements or wraps a few machine learning tools Scikit-Learn style."

setup(
    name='SKSurrogate',
    version='0.0.5',
    author='Mehdi Ghasemi',
    author_email='mehdi.ghasemi@gmail.com',
    packages=['SKSurrogate'],
    url='https://github.com/mghasemi/sksurrogate.git',
    license='MIT License',
    description=Description,
    long_description=open('./readme.rst').read(),
    keywords=["Auto Machine Learning", "AutoML", "Optimization", "Surrogate Optimization", "Numerical",
              "Machine Learning", "Regression", "Random Search"],
    install_requires=['numpy', 'scipy', 'pandas', 'matplotlib', 'scikit-learn', 'SALib', 'imbalanced-learn',
                      'eli5', 'skrebate', 'tqdm', 'peewee', 'category_encoders']
)
