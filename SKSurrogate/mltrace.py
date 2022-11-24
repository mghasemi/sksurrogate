"""
MLTrace: A machine learning progress tracker
====================================================

This module provides some basic functionality to track the process of machine learning model development.
It sets up a SQLite db-file and stores selected models, graphs, and data (for convenience) and recovers them
as requested.

``mltrace`` uses `peewee <http://docs.peewee-orm.com/en/latest/>`_ and `pandas <https://pandas.pydata.org/>`_ for
data manipulation.

It also has built in capabilities to generate some typical plots and graph in machine learning.
"""

import numpy
import joblib

try:
    from peewee import *
except ModuleNotFoundError:
    Model = type("Model", (object,), dict(Simple=lambda: 0.0))
    SqliteDatabase = lambda x: None
from datetime import datetime

MLTRACK_DB = SqliteDatabase(None)


class np2df(object):
    """
    A class to convert numpy ndarray to a pandas DataFrame. It produces a callable object which returns a
    `pandas.DataFrame`

    :param data: `numpy.ndarray` data
    :param clmns: a list of titles for pandas DataFrame column names.

        If None, it produces `C{num}` where `num` changes as the index of the ndarray does.
    """

    def __init__(self, data, clmns=None):
        self.data = data
        self.N = len(data[0])
        if clmns is None:
            self.Columns = ["C%d" % (_) for _ in range(self.N)]
        else:
            self.Columns = clmns

    def __call__(self, *args, **kwargs):
        from pandas import DataFrame

        dct = {}
        for idx in range(self.N):
            dct[self.Columns[idx]] = list(self.data[:, idx])
        return DataFrame(dct)


class Task(Model):
    """
    The class to generate the 'task` table in the SQLite db-file.
    This table keeps basic information about the task on hand, e.g., the task name, a brief description,
    target column, and columns to be ignored.
    """

    try:
        task_id = IntegerField(primary_key=True, unique=True, null=False, default=1)
        name = CharField(null=True)
        description = TextField(null=True)
        target = CharField(null=True)
        ignore = CharField(null=True)
        init_date = DateTimeField(default=datetime.now, null=True)
        last_mod_date = DateTimeField(default=datetime.now, null=True)
    except:
        pass

    class Meta:
        database = MLTRACK_DB


class MLModel(Model):
    """
    The class to generate the 'mlmodel` table in the SQLite db-file.
    It stores the scikit-learn scheme of the model/pipeline, its parameters, etc.
    """

    try:
        model_id = IntegerField(primary_key=True, unique=True, null=False)
        task_id = ForeignKeyField(Task)
        name = CharField(null=True)
        model_str = TextField(null=True)
        model_type = CharField(null=True)
        parameters = BareField(null=True)
        date_modified = DateTimeField(default=datetime.now, null=True)
    except:
        pass

    class Meta:
        database = MLTRACK_DB


class Metrics(Model):
    """
    The class to generate the 'metrics` table in the SQLite db-file.
    This table stores the calculated metrics of each stored model.
    """

    try:
        metrics_id = IntegerField(primary_key=True, unique=True, null=False)
        model_id = ForeignKeyField(MLModel)
        accuracy = FloatField(null=True)
        auc = FloatField(null=True)
        precision = FloatField(null=True)
        recall = FloatField(null=True)
        f1 = FloatField(null=True)
        mcc = FloatField(null=True)
        logloss = FloatField(null=True)
        variance = FloatField(null=True)
        max_error = FloatField(null=True)
        mse = FloatField(null=True)
        mae = FloatField(null=True)
        r2 = FloatField(null=True)
    except:
        pass

    class Meta:
        database = MLTRACK_DB


class Saved(Model):
    """
    The class to generate the 'saved` table in the SQLite db-file.
    It keeps the pickled version of a stored model that can be later recovered.
    """

    try:
        pickle_id = IntegerField(primary_key=True, unique=True, null=False)
        model_id = ForeignKeyField(MLModel)
        pickle = BareField(null=True)
        init_date = DateTimeField(default=datetime.now, null=True)
    except:
        pass

    class Meta:
        database = MLTRACK_DB


class Plots(Model):
    """
    The class to generate the 'plots` table in the SQLite db-file.
    This table stores `matplotlib` plots associated to each model.
    """

    try:
        plot_id = IntegerField(primary_key=True, unique=True, null=False)
        model_id = ForeignKeyField(MLModel)
        title = CharField(null=True)
        plot = BareField(null=True)
        init_date = DateTimeField(default=datetime.now, null=True)
    except:
        pass

    class Meta:
        database = MLTRACK_DB


class Data(Model):
    """
    The class to generate the 'data` table in the SQLite db-file.
    This table stores the whole given data for convenience.
    """

    class Meta:
        database = MLTRACK_DB


class Weights(Model):
    """
    The class to generate the 'weights` table in the SQLite db-file.
    Stores some sensitivity measures, correlations, etc.
    """

    class Meta:
        database = MLTRACK_DB


class mltrack(object):
    """
    This class instantiates an object that tracks the ML activities and store them upon request.

    :param task: 'str' the task name
    :param task_id: the id of an existing task, if the name is not provided.
    :param db_name: a file name for the SQLite database
    :param cv: the default cross validation method, must be a valid cv based on `sklearn.model_selection`;
            default: `ShuffleSplit(n_splits=3, test_size=.25)`
    :param encode: whether to preprocess the data automatically or not;
            default: `False`
    """

    def __init__(self, task, task_id=None, db_name="mltrack.db", cv=None, encode=False):
        self.db_name = db_name
        tables = [Task, MLModel, Metrics, Saved, Plots, Data, Weights]
        for tbl in tables:
            tbl._meta.database.init(self.db_name)
        MLTRACK_DB.create_tables(tables)
        res = Task.select().where((Task.name == task) | (Task.task_id == task_id))
        if len(res) > 0:
            self.task = res[0].name
            self.task_id = res[0].task_id
            self.target = res[0].target
        else:
            new_task = Task.create(name=task, description="Initiated automatically")
            self.task_id = new_task.task_id
        import sqlite3

        self.conn = sqlite3.connect(self.db_name)
        if cv is None:
            from sklearn.model_selection import ShuffleSplit

            self.cv = ShuffleSplit(n_splits=3, test_size=0.25)
        else:
            self.cv = cv
        self.X, self.y = None, None
        self.encode = encode
        self.Updated, self.Loaded, self.Recovered = [], [], []

    def UpdateTask(self, data):
        """
        Updates the current task info.

        :param data: a dictionary that may include some the followings as its keys:

                + 'name': the corresponding value will replace the current task name

                + 'description': the corresponding value will replace the current description

                + 'ignore': the corresponding value will replace the current ignored columns
        :return: None
        """
        task = Task.select().where(Task.task_id == self.task_id).get()
        if "name" in data:
            task.name = data["name"]
        if "description" in data:
            task.description = data["description"]
        if "ignore" in data:
            task.ignore = ",".join(data["ignore"])
        task.last_mod_date = datetime.now()
        task.save()

    def UpdateModel(self, mdl, name):
        """
        Updates an already logged model which has `mltrack_id` set.

        :param mdl: a scikit-learn compatible estimator/pipeline
        :param name: an arbitrary string to name the model
        :return:  None
        """
        from pickle import dumps

        if "mltrack_id" not in mdl.__dict__:
            return
        else:
            mltrack_id = mdl.mltrack_id
        model = MLModel.select().where(MLModel.model_id == mltrack_id).get()
        model.name = name
        model.model_str = str(mdl)
        model.parameters = dumps(mdl.get_params())
        model.date_modified = datetime.now()
        model.save()
        if mltrack_id not in self.Updated:
            self.Updated.append(mltrack_id)

    def LogModel(self, mdl, name=None):
        """
        Log a machine learning model

        :param mdl: a scikit-learn compatible estimator/pipeline
        :param name: an arbitrary string to name the model
        :return: modified instance of `mdl` which carries a new attribute `mltrack_id` as its id.
        """
        from pickle import dumps

        if name is not None:
            setattr(mdl, 'mltrack_name', name)
        else:
            setattr(mdl, 'mltrack_name', name if name is not None else str(mdl).split("(")[0])
        if "mltrack_id" not in mdl.__dict__:
            MLModel.create(
                task_id=self.task_id,
                name=mdl.mltrack_name,
                model_str=str(mdl),
                model_type=str(type(mdl)).split("'")[1],
                parameters=dumps(mdl.get_params()),
            )
            mdl.mltrack_id = (
                MLModel.select(MLModel.model_id).order_by(MLModel.model_id.desc()).get()
            )
        else:
            res = MLModel.select().where(MLModel.model_id == mdl.mltrack_id)[0]
            res.name = mdl.mltrack_name
            res.model_str = str(mdl)
            res.parameters = dumps(mdl.get_params())
            res.date_modified = datetime.now()
            res.save()
            # TBM
            Tskres = Task.select().where(Task.task_id == self.task_id)[0]
            Tskres.last_mod_date = datetime.now()
            Tskres.save()
        return mdl

    def RegisterData(self, source_df, target):
        """
        Registers a pandas DataFrame into the SQLite database.
        Upon a call, it also sets `self.X` and `self.y` which are numpy arrays.

        :param source_df: the pandas DataFrame to be stored
        :param target: the name of the target column to be predicted
        :return: None
        """
        # TBM
        res = Task.select().where(Task.task_id == self.task_id)[0]
        res.target = target
        res.last_mod_date = datetime.now()
        res.save()
        self.target = target
        if self.encode:
            # TODO: Input parameters for DataPreprocess
            from .DataProcess import DataPreprocess
            encoder = DataPreprocess(source_df)
            encoder.encode()
            source_df = encoder.transformed_df
        clmns = list(source_df.columns)
        if target not in clmns:
            raise BaseException("`%s` is not a part of data source." % target)
        source_df.to_sql("data", self.conn, if_exists="replace", index=False)
        clmns.remove(target)
        self.X = source_df[clmns].values
        self.y = source_df[target].values

    def get_data(self):
        """
        Retrieves data in numpy format

        :return: numpy arrays X, y
        """
        from pandas import read_sql

        df = read_sql("SELECT * FROM data", self.conn)
        clmns = list(df.columns)
        clmns.remove(self.target)
        self.X = df[clmns].values
        self.y = df[self.target].values
        return self.X, self.y

    def get_dataframe(self):
        """
        Retrieves data in pandas DataFrame format

        :return: pandas DataFrame containing all data
        """
        from pandas import read_sql

        df = read_sql("SELECT * FROM data", self.conn)
        return df

    def LogMetrics(self, mdl, cv=None):
        """
        Logs metrics of an already logged model using a cross validation method

        :param mdl: the model to be measured
        :param cv: cross validation method
        :return: a dictionary of all measures with their corresponding values for the model
        """
        if cv is not None:
            self.cv = cv
        if self.X is None:
            self.get_data()
        if "mltrack_id" not in mdl.__dict__:
            mdl = self.LogModel(mdl)
        mdl_id = mdl.mltrack_id
        mdl_type = mdl._estimator_type
        #######################################################
        prds = []
        prbs = []
        for train_idx, test_idx in self.cv.split(self.X, self.y):
            X_train, y_train = self.X[train_idx], self.y[train_idx]
            X_test, y_test = self.X[test_idx], self.y[test_idx]
            mdl.fit(X_train, y_train)
            prds.append((mdl.predict(X_test), y_test))
            try:
                prbs.append(mdl.predict_proba(X_test)[:, 1])
            except AttributeError:
                try:
                    prbs.append(mdl.decision_function(X_test))
                except AttributeError:
                    pass
        #######################################################
        acc = None
        f_1 = None
        prs = None
        rcl = None
        aur = None
        mcc = None
        lgl = None
        vrn = None
        mxe = None
        mse = None
        mae = None
        r2 = None
        n_ = float(len(prbs))
        if mdl_type == "classifier":
            from sklearn.metrics import (
                accuracy_score,
                f1_score,
                precision_score,
                recall_score,
                roc_curve,
                auc,
                log_loss,
                matthews_corrcoef,
            )

            acc = sum([accuracy_score(y_tst, y_prd) for y_prd, y_tst in prds]) / n_
            f_1 = sum([f1_score(y_tst, y_prd, average='weighted') for y_prd, y_tst in prds]) / n_
            prs = sum([precision_score(y_tst, y_prd, average='weighted') for y_prd, y_tst in prds]) / n_
            rcl = sum([recall_score(y_tst, y_prd, average='weighted') for y_prd, y_tst in prds]) / n_
            mcc = sum([matthews_corrcoef(y_tst, y_prd) for y_prd, y_tst in prds]) / n_
            lgl = 0
            n_drops = 0
            for y_prd, y_tst in prds:
                try:
                    lgl += log_loss(y_tst, y_prd, labels=numpy.unique(y_tst))
                except ValueError:
                    n_drops += 1
            lgl = lgl / max(n_ - n_drops, 1)
            aur = 0.0
            try:
                for i in range(int(n_)):
                    fpr, tpr, _ = roc_curve(prds[i][1], prbs[i])
                    aur += auc(fpr, tpr)
            except ValueError:
                aur = 0
            aur /= n_
        elif mdl_type == "regressor":
            from sklearn.metrics import (
                explained_variance_score,
                median_absolute_error,
                mean_squared_error,
                mean_absolute_error,
                r2_score,
            )

            vrn = (
                    sum([explained_variance_score(y_tst, y_prd) for y_prd, y_tst in prds])
                    / n_
            )
            mxe = (
                    sum([median_absolute_error(y_tst, y_prd) for y_prd, y_tst in prds]) / n_
            )
            mse = sum([mean_squared_error(y_tst, y_prd) for y_prd, y_tst in prds]) / n_
            mae = sum([mean_absolute_error(y_tst, y_prd) for y_prd, y_tst in prds]) / n_
            r2 = sum([r2_score(y_tst, y_prd) for y_prd, y_tst in prds]) / n_
        Metrics.create(
            model_id=mdl_id,
            accuracy=acc,
            auc=aur,
            precision=prs,
            f1=f_1,
            recall=rcl,
            mcc=mcc,
            logloss=lgl,
            variance=vrn,
            max_error=mxe,
            mse=mse,
            mae=mae,
            r2=r2,
        )
        # TBM
        res = Task.select().where(Task.task_id == self.task_id)[0]
        res.last_mod_date = datetime.now()
        res.save()
        return dict(
            accuracy=acc,
            auc=aur,
            precision=prs,
            f1=f_1,
            recall=rcl,
            mcc=mcc,
            logloss=lgl,
            variance=vrn,
            max_error=mxe,
            mse=mse,
            mae=mae,
            r2=r2,
        )

    def LoadModel(self, mid):
        """
        Loads a model corresponding to an id

        :param mid: the model id
        :return: an unfitted model
        """
        from importlib import import_module
        from pickle import loads

        res = MLModel.select().where(MLModel.model_id == mid)
        if len(res) == 0:
            raise BaseException("No model with id '%d' were found" % (mid))
        detail = res[0].model_type.split(".")
        module_str = ".".join(detail[:-1])
        clss = detail[-1]
        module = import_module(module_str)
        params = loads(res[0].parameters)
        mdl = module.__getattribute__(clss)()
        mdl.set_params(**params)
        mdl.mltrack_id = mid
        if mid not in self.Loaded:
            self.Loaded.append(mid)
        return mdl

    @staticmethod
    def getBest(metric):
        """
        Finds the model with the best metric.

        :param metric: the metric to find the best stored model for
        :return: the model wiith the best `metric`
        """
        res = (
            Metrics.select()
            .order_by(Metrics.__dict__[metric].__dict__["field"].desc())
            .dicts()
        )
        return res[0]

    def allTasks(self):
        """
        Lists all tasks as a pandas DataFrame

        :return: a pandas DataFrame
        """
        from pandas import read_sql

        return read_sql("SELECT * FROM task", self.conn)

    def allModels(self):
        """
        Lists all logged models as a pandas DataFrame

        :return: a pandas DataFrame
        """
        from pandas import read_sql

        return read_sql(
            "SELECT model_id, task_id, name, model_str, model_type, date_modified FROM mlmodel WHERE task_id=%d"
            % (self.task_id),
            self.conn,
        )

    def allPreserved(self):
        """
        Lists all pickled models as a pandas DataFrame

        :return: a pandas DataFrame
        """
        from pandas import read_sql

        return read_sql("SELECT pickle_id, model_id, init_date FROM saved", self.conn)

    def PreserveModel(self, mdl):
        """
        Pickles and preserves an already logged model

        :param mdl: a logged model
        :return: None
        """

        if "mltrack_id" not in mdl.__dict__:
            mdl = self.LogModel(mdl)
        mdl_id = mdl.mltrack_id
        file = open("track_ml_tmp_mdl.joblib", "wb")
        joblib.dump(mdl, file)
        file.close()
        file = open("track_ml_tmp_mdl.joblib", "rb")
        str_cntnt = file.read()
        Saved.create(model_id=mdl_id, pickle=str_cntnt)
        file.close()
        import os

        os.remove("track_ml_tmp_mdl.joblib")

    def RecoverModel(self, mdl_id):
        """
        Recovers a pickled model

        :param mdl_id: a valid `mltrack_id`
        :return: a fitted model
        """

        res = (
            Saved.select()
            .where(Saved.model_id == mdl_id)
            .order_by(Saved.init_date.desc())
            .dicts()
        )
        file = open("track_ml_tmp_mdl.joblib", "wb")
        file.write(res[0]["pickle"])
        file.close()
        file = open("track_ml_tmp_mdl.joblib", "rb")
        mdl = joblib.load(file)
        file.close()
        import os

        os.remove("track_ml_tmp_mdl.joblib")
        if mdl_id not in self.Recovered:
            self.Recovered.append(mdl_id)
        return mdl

    def allPlots(self, mdl_id):
        """
        Lists all stored plots for a model with `mdl_id` as a pandas DataFrame

        :param mdl_id: a valid `mltrack_id`
        :return: a pandas DataFrame
        """
        from pandas import read_sql

        return read_sql(
            "SELECT plot_id, model_id, title, init_date FROM plots WHERE model_id=%d"
            % (mdl_id),
            self.conn,
        )

    @staticmethod
    def LoadPlot(pid):
        """
        Loads a `matplotlib` plot

        :param pid: the id of the plot
        :return: a `matplotlib` figure
        """
        from pickle import loads

        # ax = plt.subplot(111)
        res = Plots.select().where(Plots.plot_id == pid).dicts()
        fig = loads(res[0]["plot"])
        return fig

    def plot_learning_curve(
            self, mdl, title, ylim=None, cv=None, n_jobs=1, train_sizes=None, **kwargs
    ):
        """
        Generate a simple plot of the test and training learning curve.

        :param mdl: object type that implements the "fit" and "predict" methods;
            An object of that type which is cloned for each validation.
        :param title: string;
            Title for the chart.
        :param measure: string, a performance measure; must be one of hte followings:
            `accuracy`, `f1`, `precision`, `recall`, `roc_auc`
        :param ylim: tuple, shape (ymin, ymax), optional;
            Defines minimum and maximum yvalues plotted.
        :param cv: int, cross-validation generator or an iterable, optional;
            Determines the cross-validation splitting strategy.
            Possible inputs for cv are:

              - None, to use the default 3-fold cross-validation,

              - integer, to specify the number of folds.

              - An object to be used as a cross-validation generator.

              - An iterable yielding train/test splits.

            For integer/None inputs, if ``y`` is binary or multiclass,
            :class:`StratifiedKFold` used. If the mdl is not a classifier
            or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.
        :param n_jobs: integer, optional;
            Number of jobs to run in parallel (default 1).
        :return: a `matplotlib` plot
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from sklearn.model_selection import learning_curve

        if cv is not None:
            self.cv = cv

        if self.X is None:
            self.get_data()

        if "mltrack_id" not in mdl.__dict__:
            mdl = self.LogModel(mdl)
        mdl_id = mdl.mltrack_id

        meas = kwargs.get("measure", "accuracy")
        if meas not in ["accuracy", "f1", "precision", "recall", "roc_auc"]:
            meas = "accuracy"
        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 5)
        plt.subplot(111)
        fig = plt.figure()
        plt.title(title)
        if ylim is None:
            ylim = (-0.05, 1.05)
        plt.ylim(*ylim)
        plt.xlabel("Training size")
        plt.ylabel("Score (%s)" % (meas))
        train_sizes, train_scores, test_scores = learning_curve(
            mdl,
            self.X,
            self.y,
            cv=self.cv,
            n_jobs=n_jobs,
            train_sizes=train_sizes,
            scoring=meas,
        )
        xlbls = np.array(
            [str(round(_ * 100, 1)) + " %" for _ in train_sizes / len(self.y)]
        )
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        plt.grid()

        plt.fill_between(
            xlbls,
            train_scores_mean - train_scores_std,
            train_scores_mean + train_scores_std,
            alpha=0.1,
            color="r",
        )
        plt.fill_between(
            xlbls,
            test_scores_mean - test_scores_std,
            test_scores_mean + test_scores_std,
            alpha=0.1,
            color="g",
        )
        plt.plot(xlbls, train_scores_mean, "o-", color="r", label="Training score")
        plt.plot(
            xlbls, test_scores_mean, "o-", color="g", label="Cross-validation score"
        )
        plt.legend(loc="best")

        from pickle import dumps

        pckl = dumps(fig)
        Plots.create(model_id=mdl_id, title=meas, plot=pckl)

        return plt

    def split_train(self, mdl):
        from sklearn.model_selection import train_test_split

        if "mltrack_id" not in mdl.__dict__:
            mdl = self.LogModel(mdl)
        mdl_id = mdl.mltrack_id

        if self.X is None:
            self.get_data()

        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, train_size=0.75
        )

        from sklearn.exceptions import NotFittedError

        x_ = X_test[0]
        try:
            mdl.predict([x_])
        except NotFittedError as _:
            mdl.fit(X_train, y_train)

        return mdl, mdl_id, X_train, X_test, y_train, y_test

    def plot_calibration_curve(self, mdl, name, fig_index=1, bins=10):
        """
        Plots calibration curves.

        :param mdl: object type that implements the "fit" and "predict" methods;
            An object of that type which is cloned for each validation.
        :param name: string;
            Title for the chart.
        :param bins: number of bins to partition samples
        :return: a `matplotlib` plot
        """
        import matplotlib.pyplot as plt
        from sklearn.calibration import calibration_curve

        fig = plt.figure(fig_index, figsize=(10, 10))
        ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
        ax2 = plt.subplot2grid((3, 1), (2, 0))

        ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

        mdl, mdl_id, _, X_test, _, y_test = self.split_train(mdl)

        if hasattr(mdl, "predict_proba"):
            prob_pos = mdl.predict_proba(X_test)[:, 1]
        else:  # use decision function
            prob_pos = mdl.decision_function(X_test)
            prob_pos = (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())

        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_test, prob_pos, n_bins=bins
        )

        ax1.plot(mean_predicted_value, fraction_of_positives, "s-", label="%s" % (name))

        ax2.hist(prob_pos, range=(0, 1), bins=bins, label=name, histtype="step", lw=2)

        ax1.set_ylabel("Fraction of positives")
        ax1.set_ylim([-0.05, 1.05])
        ax1.legend(loc="lower right")
        ax1.set_title("Calibration plots  (reliability curve)")

        ax2.set_xlabel("Mean predicted value")
        ax2.set_ylabel("Count")
        ax2.legend(loc="upper center", ncol=2)

        plt.tight_layout()

        from pickle import dumps

        pckl = dumps(fig)
        Plots.create(model_id=mdl_id, title="calibration", plot=pckl)

        return plt

    def plot_roc_curve(self, mdl, label=None):
        """
        The ROC curve, modified from Hands-On Machine learning with Scikit-Learn.

        :param mdl: object type that implements the "fit" and "predict" methods;
            An object of that type which is cloned for each validation.
        :param label: string;
            label for the chart.
        :return: a `matplotlib` plot
        """
        import matplotlib.pyplot as plt
        from numpy import arange
        from sklearn.metrics import roc_curve

        mdl, mdl_id, _, X_test, _, y_test = self.split_train(mdl)

        _ = plt.subplot(111)
        fig = plt.figure(figsize=(8, 8))
        plt.title("ROC Curve")
        try:
            y_score = mdl.predict_proba(X_test)[:, 1]
        except:
            y_score_ = mdl.decision_function(X_test)
            y_score = (y_score_ - y_score_.min()) / (y_score_.max() - y_score_.min())
        fpr, tpr, _ = roc_curve(y_test, y_score)
        plt.plot(fpr, tpr, linewidth=2, label=label)
        plt.plot([0, 1], [0, 1], "k--")
        plt.axis([-0.005, 1, 0, 1.005])
        plt.xticks(arange(0, 1, 0.05), rotation=90)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate (Recall)")
        plt.legend(loc="best")

        from pickle import dumps

        pckl = dumps(fig)
        Plots.create(model_id=mdl_id, title="roc curve", plot=pckl)

        return plt

    def plot_cumulative_gain(
            self,
            mdl,
            title="Cumulative Gains Curve",
            figsize=None,
            title_fontsize="large",
            text_fontsize="medium",
    ):
        """
        Generates the Cumulative Gains Plot from labels and scores/probabilities
        The cumulative gains chart is used to determine the effectiveness of a
        binary classifier. A detailed explanation can be found at
        `http://mlwiki.org/index.php/Cumulative_Gain_Chart <http://mlwiki.org/index.php/Cumulative_Gain_Chart>`_.
        The implementation here works only for binary classification.

        :param mdl: object type that implements the "fit" and "predict" methods;

            An object of that type which is cloned for each validation.
        :param title: (string, optional): Title of the generated plot.

            Defaults to "Cumulative Gains Curve".
        :param figsize: (2-tuple, optional): Tuple denoting figure size of the plot e.g. (6, 6).

            Defaults to ``None``.
        :param title_fontsize: (string or int, optional): Matplotlib-style fontsizes.

            Use e.g., "small", "medium", "large" or integer-values. Defaults to "large".
        :param text_fontsize: (string or int, optional): Matplotlib-style fontsizes.

            Use e.g. "small", "medium", "large" or integer-values. Defaults to "medium".
        :return: ax (:class:`matplotlib.axes.Axes`): The axes on which the plot was drawn.
        """
        from numpy import array, unique
        import matplotlib.pyplot as plt

        mdl, mdl_id, _, X_test, _, y_test = self.split_train(mdl)

        y_true = array(y_test)
        try:
            y_probas = mdl.predict_proba(X_test)
            y_probas = array(y_probas)
            prob_pos0 = y_probas[:, 0]
            prob_pos1 = y_probas[:, 1]
        except:
            prob_pos = mdl.decision_function(X_test)
            prob_pos1 = (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
            prob_pos0 = (prob_pos.max() - prob_pos) / (prob_pos.max() - prob_pos.min())

        classes = unique(y_true)
        if len(classes) != 2:
            raise ValueError(
                "Cannot calculate Cumulative Gains for data with "
                "{} category/ies".format(len(classes))
            )

        # Compute Cumulative Gain Curves
        percentages, gains1 = self.cumulative_gain_curve(y_true, prob_pos0, classes[0])
        percentages, gains2 = self.cumulative_gain_curve(y_true, prob_pos1, classes[1])

        fig, ax = plt.subplots(1, 1, figsize=figsize)

        ax.set_title(title, fontsize=title_fontsize)

        ax.plot(percentages, gains1, lw=3, label="Class {}".format(classes[0]))
        ax.plot(percentages, gains2, lw=3, label="Class {}".format(classes[1]))

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.0])

        ax.plot([0, 1], [0, 1], "k--", lw=2, label="Baseline")

        ax.set_xlabel("Percentage of sample", fontsize=text_fontsize)
        ax.set_ylabel("Gain", fontsize=text_fontsize)
        ax.tick_params(labelsize=text_fontsize)
        ax.grid(True)
        ax.legend(loc="lower right", fontsize=text_fontsize)

        from pickle import dumps

        pckl = dumps(fig)
        Plots.create(model_id=mdl_id, title="cumulative gain", plot=pckl)

        return ax

    @staticmethod
    def cumulative_gain_curve(y_true, y_score, pos_label=None):
        """
        This function generates the points necessary to plot the Cumulative Gain
        Note: This implementation is restricted to the binary classification task.

        :param y_true: (array-like, shape (n_samples)): True labels of the data.
        :param y_score: (array-like, shape (n_samples)): Target scores, can either be probability estimates of
            the positive class, confidence values, or non-thresholded measure of decisions (as returned by
            decision_function on some classifiers).
        :param pos_label: (int or str, default=None): Label considered as positive and others are considered negative
        :return:
            percentages (numpy.ndarray): An array containing the X-axis values for plotting the Cumulative Gains chart.
            gains (numpy.ndarray): An array containing the Y-axis values for one curve of the Cumulative Gains chart.
        :raise:
            ValueError: If `y_true` is not composed of 2 classes. The Cumulative Gain Chart is only relevant in
            binary classification.
        """
        from numpy import asarray, array_equal, cumsum, arange, insert, unique, argsort

        y_true, y_score = asarray(y_true), asarray(y_score)

        # ensure binary classification if pos_label is not specified
        classes = unique(y_true)
        if pos_label is None and not (
                array_equal(classes, [0, 1])
                or array_equal(classes, [-1, 1])
                or array_equal(classes, [0])
                or array_equal(classes, [-1])
                or array_equal(classes, [1])
        ):
            raise ValueError("Data is not binary and pos_label is not specified")
        elif pos_label is None:
            pos_label = 1.0

        # make y_true a boolean vector
        y_true = y_true == pos_label

        sorted_indices = argsort(y_score)[::-1]
        y_true = y_true[sorted_indices]
        gains = cumsum(y_true)

        percentages = arange(start=1, stop=len(y_true) + 1)

        gains = gains / float(sum(y_true))
        percentages = percentages / float(len(y_true))

        gains = insert(gains, 0, [0])
        percentages = insert(percentages, 0, [0])

        return percentages, gains

    def plot_lift_curve(
            self,
            mdl,
            title="Lift Curve",
            figsize=None,
            title_fontsize="large",
            text_fontsize="medium",
    ):
        """
        Generates the Lift Curve from labels and scores/probabilities The lift curve is used to
        determine the effectiveness of a binary classifier. A detailed explanation can be found at
        `http://www2.cs.uregina.ca/~dbd/cs831/notes/lift_chart/lift_chart.html <http://www2.cs.uregina.ca/~dbd/cs831/notes/lift_chart/lift_chart.html>`_.
        The implementation here works only for binary classification.

        :param mdl: object type that implements the "fit" and "predict" methods;
            An object of that type which is cloned for each validation.
        :param title: (string, optional): Title of the generated plot. Defaults to "Lift Curve".
        :param figsize: (2-tuple, optional): Tuple denoting figure size of the plot e.g. (6, 6). Defaults to ``None``.
        :param title_fontsize: (string or int, optional): Matplotlib-style fontsizes. Use e.g. "small", "medium",
            "large" or integer-values. Defaults to "large".
        :param text_fontsize: (string or int, optional): Matplotlib-style fontsizes. Use e.g. "small", "medium",
            "large" or integer-values. Defaults to "medium".
        :return: ax (:class:`matplotlib.axes.Axes`): The axes on which the plot was drawn.
        """
        import matplotlib.pyplot as plt
        from numpy import array, unique

        mdl, mdl_id, _, X_test, _, y_test = self.split_train(mdl)

        y_true = array(y_test)
        try:
            y_probas = mdl.predict_proba(X_test)
            y_probas = array(y_probas)
            prob_pos0 = y_probas[:, 0]
            prob_pos1 = y_probas[:, 1]
        except:
            prob_pos = mdl.decision_function(X_test)
            prob_pos1 = (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
            prob_pos0 = (prob_pos.max() - prob_pos) / (prob_pos.max() - prob_pos.min())

        classes = unique(y_true)
        if len(classes) != 2:
            raise ValueError(
                "Cannot calculate Lift Curve for data with "
                "{} category/ies".format(len(classes))
            )

        # Compute Cumulative Gain Curves
        percentages, gains1 = self.cumulative_gain_curve(y_true, prob_pos0, classes[0])
        percentages, gains2 = self.cumulative_gain_curve(y_true, prob_pos1, classes[1])

        percentages = percentages[1:]
        gains1 = gains1[1:]
        gains2 = gains2[1:]

        gains1 = gains1 / percentages
        gains2 = gains2 / percentages

        fig, ax = plt.subplots(1, 1, figsize=figsize)

        ax.set_title(title, fontsize=title_fontsize)

        ax.plot(percentages, gains1, lw=3, label="Class {}".format(classes[0]))
        ax.plot(percentages, gains2, lw=3, label="Class {}".format(classes[1]))

        ax.plot([0, 1], [1, 1], "k--", lw=2, label="Baseline")

        ax.set_xlabel("Percentage of sample", fontsize=text_fontsize)
        ax.set_ylabel("Lift", fontsize=text_fontsize)
        ax.tick_params(labelsize=text_fontsize)
        ax.grid(True)
        ax.legend(loc="lower right", fontsize=text_fontsize)

        from pickle import dumps

        pckl = dumps(fig)
        Plots.create(model_id=mdl_id, title="lift curve", plot=pckl)

        return ax

    def heatmap(
            self,
            corr_df=None,
            sort_by=None,
            ascending=False,
            font_size=3,
            cmap="gnuplot2",
            idx_col="feature",
            ignore=(),
    ):
        """
        Plots a heatmap from the values of the dataframe `corr_df`

        :param corr_df: value container
        :param idx_col: the column whose values will be used as index
        :param sort_by: dataframe will be sorted descending by values of this column.

            If None, the first column is used
        :param font_size: font size, defalut 3
        :param cmap: color mapping. Must be one of the followings

            'viridis', 'plasma', 'inferno', 'magma', 'cividis', 'Greys', 'Purples',

            'Blues', 'Greens', 'Oranges', 'Reds', 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd',

            'RdPu', 'BuPu', 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn',

            'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink', 'spring',

            'summer', 'autumn', 'winter', 'cool', 'Wistia', 'hot', 'afmhot',

            'gist_heat', 'copper', 'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',

            'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic', 'twilight',

            'twilight_shifted', 'hsv', 'Pastel1', 'Pastel2', 'Paired', 'Accent',

            'Dark2', 'Set1', 'Set2', 'Set3', 'tab10', 'tab20', 'tab20b', 'tab20c',

            'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern', 'gnuplot',

            'gnuplot2', 'CMRmap', 'cubehelix', 'brg', 'gist_rainbow', 'rainbow',

            'jet', 'nipy_spectral', 'gist_ncar'
        :return: matplotlib pyplot instance
        """
        import matplotlib.pyplot as plt
        from numpy import arange, amin, amax
        from pandas import read_sql

        ax = plt.gca()
        idx_col = idx_col
        if corr_df is None:
            df = read_sql("SELECT * FROM weights", self.conn)
            clmns = list(df.columns)
            df = df.sort_values(
                by=clmns[0] if sort_by is None else sort_by, ascending=ascending
            )
            if idx_col is None:
                idx_col = clmns[0]
            clmns.remove(idx_col)
        else:
            df = corr_df
            clmns = list(df.columns)
            # df = df.sort_values(by=clmns[0] if sort_by is None else sort_by, ascending=ascending)
            if idx_col is not None:
                # idx_col = clmns[0]
                clmns.remove(idx_col)
        for itm in ignore:
            clmns.remove(itm)
        data = df[clmns].values
        mn, mx = amin(data), amax(data)
        im = ax.imshow(data, cmap=cmap, interpolation="bilinear")
        # ax.set_adjustable(adjustable='box', share=False)
        ax.autoscale(False)
        cbar_kw = {
            "fraction": 0.2,
            "ticks": [mn, 0.0, (mn + mx) / 2.0, mx],
            "drawedges": False,
        }
        cbar = ax.figure.colorbar(im, ax=ax, aspect=max(20, len(df)), **cbar_kw)
        cbarlabel = ""
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
        cbar.ax.tick_params(labelsize=font_size + 1)
        ax.set_xticks(arange(data.shape[1]))
        ax.set_yticks(arange(data.shape[0]))
        ax.set_xticklabels(clmns, fontdict={"fontsize": font_size})
        if idx_col is not None:
            ax.set_yticklabels(list(df[idx_col]), fontdict={"fontsize": font_size})
        else:
            ax.set_yticklabels(list(df.index), fontdict={"fontsize": font_size})
        # Let the horizontal axes labeling appear on top.
        ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
        # Rotate the tick labels and set their alignment.
        plt.setp(
            ax.get_xticklabels(),
            rotation=-305,
            ha="left",
            va="top",
            rotation_mode="anchor",
        )
        # Turn spines off and create white grid.
        for _, spine in ax.spines.items():
            spine.set_visible(False)
        ax.set_xticks(arange(data.shape[1] + 1) - 0.5, minor=True)
        ax.set_yticks(arange(data.shape[0] + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="w", linestyle="-", linewidth=0)
        ax.tick_params(which="minor", bottom=False, left=False)
        return plt

    def FeatureWeights(self, weights=("pearson", "variance"), **kwargs):
        """
        Calculates the requested weights and log them

        :param weights: a list of weights, a subset of {'pearson', 'variance', 'relieff',
            'surf', 'sobol', 'morris', 'delta_mmnt', 'info-gain'}
        :param kwargs: all input acceptable by ``skrebate.ReliefF``, ``skrebate.surf``,
            ``sensapprx.SensAprx``
        :return: None
        """
        from pandas import DataFrame, read_sql

        self.data = read_sql("SELECT * FROM data", self.conn)
        features = list(self.data.columns)
        features.remove(self.target)
        weights_df = read_sql("SELECT * FROM weights", self.conn)
        if len(weights_df) == 0:
            weights_df = DataFrame({"feature": features})
        X = self.data[features].values
        y = self.data[self.target].values
        n_features = kwargs.get("n_features", int(len(features) / 2))
        domain = None
        probs = None
        regressor = kwargs.get("regressor", None)
        reduce = kwargs.get("reduce", True)
        num_smpl = kwargs.get("num_smpl", 700)
        W = {"feature": features}
        for factor in weights:
            if factor == "pearson":
                Res = dict(self.data.corr(method="pearson").fillna(0)[self.target])
                W["pearson"] = [Res[v] for v in features]
            elif factor == "variance":
                Res = dict(self.data.var())
                W["variance"] = [Res[v] for v in features]
            elif factor == "relieff":
                from skrebate import ReliefF

                n_neighbors = kwargs.get("n_neighbors", 80)
                RF = ReliefF(n_features_to_select=n_features, n_neighbors=n_neighbors)
                RF.fit(X, y)
                W["relieff"] = [
                    RF.feature_importances_[features.index(v)] for v in features
                ]
            elif factor == "surf":
                from skrebate import SURF

                RF = SURF(n_features_to_select=n_features)
                RF.fit(X, y)
                W["surf"] = [
                    RF.feature_importances_[features.index(v)] for v in features
                ]
            elif factor == "sobol":
                from .sensapprx import SensAprx

                SF = SensAprx(
                    method="sobol",
                    domain=domain,
                    probs=probs,
                    regressor=regressor,
                    reduce=reduce,
                    num_smpl=num_smpl,
                )
                SF.fit(X, y)
                domain = SF.domain
                probs = SF.probs
                W["sobol"] = [SF.weights_[features.index(v)] for v in features]
            elif factor == "morris":
                from .sensapprx import SensAprx

                SF = SensAprx(
                    method="morris",
                    domain=domain,
                    probs=probs,
                    regressor=regressor,
                    reduce=reduce,
                    num_smpl=num_smpl,
                )
                SF.fit(X, y)
                domain = SF.domain
                probs = SF.probs
                W["morris"] = [SF.weights_[features.index(v)] for v in features]
            elif factor == "delta-mmnt":
                from .sensapprx import SensAprx

                SF = SensAprx(
                    method="delta-mmnt",
                    domain=domain,
                    probs=probs,
                    regressor=regressor,
                    reduce=reduce,
                    num_smpl=num_smpl,
                )
                SF.fit(X, y)
                domain = SF.domain
                probs = SF.probs
                W["delta_mmnt"] = [SF.weights_[features.index(v)] for v in features]
            elif factor == "info-gain":
                from sklearn.feature_selection import mutual_info_classif

                Res = mutual_info_classif(X, y, discrete_features=True)
                W["info_gain"] = [Res[features.index(v)] for v in features]
        new_w_df = DataFrame(W)
        merged = weights_df.merge(new_w_df, on="feature")
        merged.fillna(0.0)
        merged.to_sql("weights", self.conn, if_exists="replace", index=False)

    def TopFeatures(self, num=10):
        """
        Returns `num` of top features in the data based on calculated weights

        :param num: number of top features to return
        :return: an OrderedDict of top features
        """
        from pandas import read_sql
        from collections import OrderedDict

        weights_df = read_sql("SELECT * FROM weights", self.conn)
        weights = list(weights_df.columns)
        weights.remove("feature")
        fields = {}
        for w in weights:
            weights_df.sort_values(by=w, ascending=False, inplace=True)
            if w == "pearson":
                feat = list(weights_df["feature"])
                candids = set(feat[:num]).union(feat[-num:])
            else:
                feat = list(weights_df["feature"])
                candids = set(feat[:num])
            for fld in candids:
                if fld in fields:
                    fields[fld] += 1
                else:
                    fields[fld] = 1
        return OrderedDict(sorted(fields.items(), key=lambda t: t[1], reverse=True))

    def Stats(self):
        df = self.get_dataframe()[self.target]
        return dict(df.describe())
