from confidence_features import ConfidenceFeatures
from transforms import ClassifierTester

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn import metrics

import numpy as np
import pandas as pd

import argparse
import os
import sys
import pickle


class CalibratedModel():
    def __init__(self, classifier, calibration, threshold=0.75, cv_ratio=0.2):
        self.classifier = classifier
        self.calibration = calibration
        self.threshold = threshold
        self.cv_ratio = cv_ratio

    def fit(self, X, y):
        cv_size = int(self.cv_ratio*len(X))
        X_train = X[0:-cv_size, :]
        y_train = y[0:-cv_size]
        self.classifier.fit(X_train, y_train)

        X_cv = X[-cv_size:-1, :]
        y_cv = y[-cv_size:-1]
        y_pred = self.classifier.predict_proba(X_cv)
        self.calibration.fit(y_pred[:, 0:1], y_cv)

    def predict(self, X):
        return self.predict_proba(X) > self.threshold

    def predict_proba(self, X):
        yt = self.classifier.predict_proba(X)
        return self.calibration.predict(yt[:, 0:1])


def load_feature_extractor(receipts_path):
    receipts = pd.read_csv(receipts_path, index_col='id')
    extractor = ConfidenceFeatures(receipts)
    return extractor


def train_model(X, y):
    model = CalibratedModel(
        RandomForestClassifier(n_estimators=100, min_samples_leaf=20, n_jobs=-1, random_state=1),
        KNeighborsRegressor(n_neighbors=100)
    )
    model.fit(X, y)
    return model


def save_model(model, path):
    with open(path, 'w') as model_file:
        pickle.dump(model, model_file)


def load_pipeline(model_path, receipts_path):
    feature_extractor = load_feature_extractor(receipts_path)
    with open(model_path) as model_file:
        model = pickle.load(model_file)
    return Pipeline([
        ('feature_extraction', feature_extractor),
        ('model', model),
    ])


def test_models(X, y):
    classifiers = [
        RandomForestClassifier(random_state=1),
        SVC(random_state=1, probability=True),
        KNeighborsClassifier()
    ]

    threshold_list = [0.8, 0.82, 0.84, 0.86, 0.88, 0.9, 0.92]
    model = Pipeline([
        ('classifier', ClassifierTester(classifiers,
                                        predict_flag=False,
                                        proba_flag=True,
                                        threshold_lst=threshold_list))
    ])

    # we only need to do a model fit here as the ClassifierTester takes care to
    # run each classifier under consideration with the data split in training
    # and test set
    model.fit(X, y)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Train a classifier.')

    parser.add_argument('--targets', metavar='targets.csv',
                        type=str, nargs=1, default=['targets.csv'],
                        help='target values for the historic receipts')

    parser.add_argument('--historic-receipts', metavar='historic_receipts.csv',
                        type=str, nargs=1, default=['historic_receipts.csv'],
                        help='receipts with target values used for training')

    parser.add_argument('--current-receipts', metavar='receipts.csv',
                        type=str, nargs=1, default=['receipts.csv'],
                        help='receipts with correct values used for statistics')

    parser.add_argument('--count', metavar='<number>',
                        type=int, nargs=1, default=[5000],
                        help='number of receipts in training')

    parser.add_argument('--train', metavar='model.p',
                        type=str, nargs=1, default=[''],
                        help='create a trained model file')

    return parser.parse_args()


def main():
    args = parse_arguments()

    nmb_training_items = args.count[0]

    historic_receipts = pd.read_csv(args.historic_receipts[0], index_col='id')
    targets = pd.read_csv(args.targets[0], index_col='id')

    print 'Training items: ', nmb_training_items

    X_orig = historic_receipts.iloc[:nmb_training_items]
    y = targets.ix[X_orig.index, 'target_val']

    print 'Start training...'

    feature_extractor = load_feature_extractor(args.current_receipts[0])
    X = feature_extractor.transform(X_orig)

    if len(args.train[0]) == 0:
        test_models(X, y)
    else:
        model = train_model(X, y)
        save_model(model, args.train[0])


if __name__ == "__main__":
    main()
