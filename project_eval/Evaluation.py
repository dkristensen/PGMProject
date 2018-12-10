# This file contains the class to evaluate the performances of a given algorithm that generate features.

import time
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold, ParameterGrid

import json
import os


SEED = 0
METRIC_NAMES = ['Accuracy', 'Precision', 'Recall', 'F1-score']


def score_prediction(y_true, y_pred):
    """ Scores the prediction by comparing it to the truth """
    return [round(metrics.accuracy_score(y_true, y_pred)*100, 2),
            round(metrics.precision_score(y_true, y_pred)*100, 2),
            round(metrics.recall_score(y_true, y_pred)*100, 2),
            round(metrics.f1_score(y_true, y_pred)*100, 2)]


class EEGEval:

    def __init__(self):
        """
        Evaluates the performance of an algorithm on the data.

        Example:
        >>> evaluator = EEGEval()
        >>> algorithm = ...
        >>> predictor = ...
        >>> evaluator.evaluate(algorithm, predictor, cv_fold=5)
        >>> evaluator.result
        """

        self.result = None  # To store the result
        self.predictor = None  # To store the fit predictor

    @staticmethod
    def load_data_part(fname):
        """ read and prepare training data """
        if "_data" not in fname:
            return None
        # Read data
        data = pd.read_csv(fname)
        # events file
        events_fname = fname.replace('_data', '_events')
        # read event file
        labels = pd.read_csv(events_fname)
        clean = data.drop(['id'], axis=1)  # remove id
        labels = labels.drop(['id'], axis=1)  # remove id
        return clean, labels

    def generate_features(self, data_path, algorithm):
        features = dict()
        for which in ['train', 'test']:
            x = []
            y = []
            for file in os.listdir(os.path.join(data_path, which)):
                fname = os.path.join(data_path, which, file)
                loaded_part = self.load_data_part(fname)
                if loaded_part is None:
                    continue
                data, labels = loaded_part
                data = algorithm.generate_features(np.asarray(data.astype(float)))
                x.append(data)
                y.append(labels.values)
            features['x_'+which] = np.vstack(x)
            features['y_'+which] = np.vstack(y)
        return features

    def score_features(self, features, predictor, cv_fold, verbose=0):
        """
        Runs a classification task with optimum parameters.
        features has 4 keys (x_train, x_test, y_train, y_test) to use for parameter tuning.
        :param dict features: Contains the data to use with the predictor.
        :param BasePredictor predictor: The predictor for which we want to find the best parameters
        :param int cv_fold: number of folds for cross validation
        :param int verbose: degree of verbosity
        :return list of metrics applied to the prediction as well as the fit and predict time
        """
        # First we optimise the hyper parameters:
        # data has 4 keys but only 2 (x_train and y_train) will be used for the optimization
        best_params = optimize_hyper_parameters(features, predictor, cv_fold, verbose)
        predictor.set_hyper_parameters(best_params)

        # Then we fit the predictor:
        predictor.fit(features)

        # Afterwards, we generate the prediction
        y_pred = predictor.predict(features)

        # Finally, we compute the metrics:
        metric_res = score_prediction(features['y_test'], y_pred)

        self.predictor = predictor

        return metric_res, best_params

    def evaluate(self, data_path, algorithm, predictor, cv_fold=5, sub_select=1, verbose=1):
        """
        Runs the algorithm that evaluates the function
        :param data_path: path to the data containing 2 folders: train and test
        :param algorithm: Algorithm to evaluate
        :param predictor: Predictor to use for classification
        :param cv_fold: Number of fold to use for cross validation when optimizing the predictor.
        :param sub_select: type of sub-selection to use to balance the data.
        :param verbose: degree of verbosity
        :return:
        """

        # Generating features
        if verbose:
            print("Generating features...")
        t0 = time.time()
        features = self.generate_features(data_path, algorithm)
        algo_time = round(time.time() - t0, 2)
        name = algorithm.name + ' - ' + predictor.name
        y_train = features['y_train']
        y_test = features['y_test']

        # Training and finding best hyper parameters for predictor
        scores_list = []
        names_list = []
        for j in range(y_train.shape[1]):
            if verbose:
                print("Scoring {} out of {}...".format(j+1, y_train.shape[1]))
            features['y_train'] = y_train[:, j]
            features['y_test'] = y_test[:, j]
            if sub_select:
                new_features = sub_select_features(features, sub_select)
            else:
                new_features = features
            scores, best_params = self.score_features(new_features, predictor, cv_fold, verbose)
            print("Best params obtained:", best_params)
            scores_list += scores
            names_list += [metric+' '+str(j) for metric in METRIC_NAMES]

        # Formatting the result
        result_list = [algo_time] + scores_list
        table = pd.DataFrame(np.array([result_list]))
        table.index = [name]
        table.columns = ['Algo time'] + names_list
        self.result = table

    def save_json(self, path):
        with open(path, 'w') as f:
            json.dump(self.result.to_dict(orient='split'), f)

    def load(self, path):
        with open(path) as f:
            result_dict = json.load(f)
        self.result = pd.DataFrame(result_dict['data'])
        self.result.columns = result_dict['columns']
        self.result.index = result_dict['index']

    def compare_folder(self, path):
        result_list = []
        eva = EEGEval()
        for file in os.listdir(path):
            fpath = os.path.join(path, file)
            if os.path.isdir(fpath):
                continue
            eva.load(fpath)
            result_list.append(eva.result)
        self.result = pd.concat(result_list)


def optimize_hyper_parameters(data, predictor, cv_fold, verbose=0):
    """
    Looks for the best hyper parameters for the predictor with the given data.
    Data has 2 keys (x_train, y_train) to use for parameter tuning.
    :param dict data: Contains the data to use to optimize the hyper parameters
    :param BasePredictor predictor: The predictor for which we want to find the best parameters
    :param int cv_fold: number of folds for cross validation
    :param int verbose: degree of verbosity
    :return The best parameter as a dict
    """
    # Hyper parameters to explore
    hyper = predictor.hyper_parameters_grid
    regs = list(ParameterGrid(hyper))
    if len(regs) == 0:
        return {}
    if len(regs) == 1:
        return regs[0]

    # Optimization
    if verbose:
        print("Optimizing...")
    scores = []
    if cv_fold > 1:
        skf = StratifiedKFold(n_splits=cv_fold, shuffle=True, random_state=SEED)

    n_param = 0
    for reg in regs:
        n_param += 1
        if verbose > 1:
            print("Optimizing parameter {0} out of {1}...".format(n_param, len(regs)))
        predictor.set_hyper_parameters(hyper_parameters=reg)
        scores_per_reg = []

        # splitting
        if cv_fold > 1:
            for train_idx, test_idx in skf.split(data['x_train'], data['y_train']):
                # Split training data in train and dev (called test as it's more convenient)
                new_data = {'x_train': data['x_train'][train_idx], 'x_test': data['x_train'][test_idx],
                            'y_train': data['y_train'][train_idx], 'y_test': data['y_train'][test_idx]}

                # Train classifier
                predictor.fit(new_data)
                score = predictor.score(new_data)
                scores_per_reg.append(score)

        else:
            predictor.fit(data)  # No cv so we fit on the whole data and we keep the best hyper params
            score = predictor.score(data)
            scores_per_reg.append(score)

        # We only keep the mean:
        scores.append(np.mean(scores_per_reg))
        if verbose > 1:
            print("Parameters {0} yielded a score of {1}.".format(reg, scores[-1]))

    best = np.argmax(scores)  # To find the hyper parameter that yielded the best score on average.
    return regs[best]


def sub_select_features(features, strategy):
    """
    Sub selects features to deal with unbalanced data
    :param features:
    :param strategy
    :return: new balanced features
    """

    def extract_one_index(y_val):
        index_ones = []
        y_prev = 0
        start_stop = []
        if y_val[-1] == 1:
            y_val = y_val.tolist() + [0]
        for i, y in enumerate(y_val):
            if y_prev == 0 and y == 1:
                start_stop = [i]
            if y_prev == 1 and y == 0:
                start_stop.append(i)
                index_ones.append(start_stop)
            y_prev = y
        return index_ones

    def wrapper(start_stop, maxi):
        size = start_stop[1] - start_stop[0]
        bound = (size+1)//2
        return [max(0, start_stop[0]-bound), min(maxi, start_stop[1]+bound)]

    def deduce_index_to_keep(one_index, maxi):
        wrapped = [wrapper(start_stop, maxi) for start_stop in one_index]
        to_keep = [idx for idx in range(wrapped[0][0], wrapped[0][1])]
        for start_stop in wrapped[1:]:
            to_keep += [idx for idx in range(start_stop[0], start_stop[1]) if idx > to_keep[-1]]
        return to_keep

    if strategy == 0:
        new_features = features  # We do nothing

    else:
        new_features = dict()
        for which in ['train', 'test']:
            one_id = extract_one_index(features['y_'+which])
            true_idx = deduce_index_to_keep(one_id, len(features['y_'+which]))
            try:
                new_features['x_'+which] = features['x_'+which][true_idx]
                new_features['y_'+which] = features['y_'+which][true_idx]
            except IndexError as e:
                print(which)
                print(features['x_'+which].shape)
                print(features['y_'+which].shape)
                print(one_id)
                raise e

    return new_features
