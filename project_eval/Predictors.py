# Python file that contains all algorithms that can classify data
# We assume the data is always a dictionary that has 4 keys : x_train, y_train, x_test, and y_test
# Fit function only uses the keys x_train and y_train
# Predict function only uses the x_test
# Score function only uses the keys x_test and y_test


from abc import ABC, abstractmethod
import numpy as np

# For each predictor:
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier


SEED = 0


class BasePredictor(ABC):

    # Properties:

    @property
    def name(self):
        """
        Returns the name of the predictor. To be implemented by user.

        :rtype: basestring
        :return: the predictor's name
        """
        raise NotImplementedError("'name' should be defined in BasePredictor implementation.")

    @property
    def hyper_parameters_grid(self):
        """The dict mapping the predictor's hyper parameters to a list of values to be tried"""
        raise NotImplementedError("'hyper_parameters_grid' should be defined in BasePredictor implementation.")

    # Checking ---------------------------------------------------------------------------------------------

    input_schema = {'x_train': np.ndarray, 'x_test': np.ndarray, 'y_train': np.ndarray, 'y_test': np.ndarray}

    def _check_input(self, data):
        assert isinstance(data, dict), "Input is not a dict"
        for name, tpe in self.input_schema.items():
            assert name in data, "Input is missing the key {}".format(name)
            assert isinstance(data[name], tpe), "Wrong type for key {}".format(name)

    # Methods ---------------------------------------------------------------------------------------------

    @abstractmethod
    def set_hyper_parameters(self, hyper_parameters):
        """
        Sets the hyper parameters of the predictor. To be implemented by user.
        :param dict hyper_parameters: The dict mapping the predictor's hyper parameters to a value to be tried
        :rtype: BasePredictor
        :return: itself
        """
        raise NotImplementedError("'set_hyper_parameter' method should be defined in BasePredictor implementation.")

    def fit(self, data):
        """
        Main fit method which is final i.e. should not be implemented by user at implementation time but called at
        usage time. To specify the fit behaviour, please implement :func:`do_fit`.

        :param  dict data: input data
        :rtype: BaseEstimator
        :return: itself
        """
        self._check_input(data)
        self.do_fit(data)
        return self

    @abstractmethod
    def do_fit(self, data):
        """
        Main fit method to be implemented by user.

        :param dict data: input data
        :rtype: BaseEstimator
        :return: itself
        """
        raise NotImplementedError("'do_fit' method should be defined in BasePredictor implementation.")

    def predict(self, data):
        """
        Main predict method which is final i.e. should not be implemented by user at implementation time but called at
        usage time. To specify the prediction behaviour, please implement :func:`do_predict`.

        :param data data: input data
        :rtype: dict
        :return: the prediction
        """
        self._check_input(data)
        result = self.do_predict(data)
        return result

    @abstractmethod
    def do_predict(self, data):
        """
        Main predict method to be implemented by user.

        :param dict data: input data
        :rtype: dict
        :return: the prediction
        """
        raise NotImplementedError("'do_predict' method should be defined in BasePredictor implementation.")

    def score(self, data):
        """
        Main score method which is final i.e. should not be implemented by user at implementation time but called at
        usage time. To specify the prediction behaviour, please implement :func:`do_score`.

        :param dict data: input data
        :rtype: float
        :return: the accuracy score
        """
        self._check_input(data)
        result = self.do_score(data)
        return result

    @abstractmethod
    def do_score(self, data):
        """
        Main score method to be implemented by user.

        :param dict data: input data
        :rtype: float
        :return: the accuracy score
        """
        raise NotImplementedError("'do_score' method should be defined in BasePredictor implementation.")


class LogReg(BasePredictor):

    name = "LogReg"

    regs = [2**t for t in range(-2, 4, 1)]
    hyper_parameters_grid = {'C': regs}

    def __init__(self, seed=SEED):
        BasePredictor.__init__(self)
        self.seed = seed
        self.predictor = LogisticRegression(random_state=seed)

    def set_hyper_parameters(self, hyper_parameters):
        self.predictor = LogisticRegression(random_state=self.seed, **hyper_parameters)

    def do_fit(self, data):
        return self.predictor.fit(data['x_train'], data['y_train'])

    def do_predict(self, data):
        prediction = self.predictor.predict(data['x_test'])
        return prediction

    def do_score(self, data):
        return self.predictor.score(data['x_test'], data['y_test'])


class RandForest(BasePredictor):

    name = "RandForest"

    regs1 = [t for t in range(100, 201, 50)]
    hyper_parameters_grid = {'n_estimators': regs1}

    def __init__(self, seed=SEED):
        BasePredictor.__init__(self)
        self.seed = seed
        self.predictor = RandomForestClassifier(n_estimators=100, random_state=seed)

    def set_hyper_parameters(self, hyper_parameters):
        self.predictor = RandomForestClassifier(random_state=self.seed, **hyper_parameters)

    def do_fit(self, data):
        return self.predictor.fit(data['x_train'], data['y_train'])

    def do_predict(self, data):
        prediction = self.predictor.predict(data['x_test'])
        return prediction

    def do_score(self, data):
        return self.predictor.score(data['x_test'], data['y_test'])


class BaggedLogReg(BasePredictor):
    name = "BaggedLogReg"

    hyper_parameters_grid = {}

    def __init__(self, seed=SEED, hyper_params_base_pred=dict()):
        BasePredictor.__init__(self)
        self.seed = seed
        base_predictor_to_bag = LogisticRegression(random_state=seed, **hyper_params_base_pred)
        self.predictor = BaggingClassifier(base_estimator=base_predictor_to_bag, n_estimators=10,
                                           random_state=seed, n_jobs=1)

    def set_hyper_parameters(self, hyper_parameters):
        base_predictor_to_bag = LogisticRegression(random_state=self.seed, **hyper_parameters)
        self.predictor = BaggingClassifier(base_estimator=base_predictor_to_bag, n_estimators=10,
                                           random_state=self.seed, n_jobs=1)

    def do_fit(self, data):
        return self.predictor.fit(data['x_train'], data['y_train'])

    def do_predict(self, data):
        prediction = self.predictor.predict(data['x_test'])
        return prediction

    def do_score(self, data):
        return self.predictor.score(data['x_test'], data['y_test'])


class BoostedLogReg(BasePredictor):
    name = "BoostedLogR"

    regs = [0.5, 1]
    hyper_parameters_grid = {'learning_rate': regs}

    def __init__(self, seed=SEED, hyper_params_base_pred=dict()):
        BasePredictor.__init__(self)
        self.seed = seed
        base_predictor_to_bag = LogisticRegression(random_state=seed, **hyper_params_base_pred)
        self.predictor = AdaBoostClassifier(base_estimator=base_predictor_to_bag, n_estimators=10,
                                            random_state=seed, learning_rate=1)

    def set_hyper_parameters(self, hyper_parameters):
        base_predictor_to_bag = LogisticRegression(random_state=self.seed)
        self.predictor = AdaBoostClassifier(base_estimator=base_predictor_to_bag, n_estimators=10,
                                            random_state=self.seed, **hyper_parameters)

    def do_fit(self, data):
        return self.predictor.fit(data['x_train'], data['y_train'])

    def do_predict(self, data):
        prediction = self.predictor.predict(data['x_test'])
        return prediction

    def do_score(self, data):
        return self.predictor.score(data['x_test'], data['y_test'])
