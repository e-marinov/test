import numpy as np
import pandas as pd
import itertools
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split
from sklearn import metrics

from sklearn.feature_selection import SelectKBest, chi2, f_classif

SCORE_FUNC = chi2


class FeatureTransform:
    """ Apply a feature mapper as a transformation.

    This is a scikit-learn style transformer that assumes the input is a
    list of parser objects. The result expands to a feature matrix containing
    one line per valid token, where the feature mapper is used to determine
    the valid tokens.

    Attributes
    ----------
    binary_targets: 1-d array
        If 'transform' is called with valid targets (one per parser), this is
        an expanded set of binary targets generated by the feature mapper for
        each token.

    values: 1-d array
        The values of valid tokens.
    """
    def __init__(self, mapper_cls):
        self.mapper = mapper_cls()

    def fit(self, X, y=None):
        pass

    def transform(self,
                  parsers,
                  currency_predictions,
                  currency_nonzero_pred,
                  y=None):
        if y is None:
            y = itertools.repeat(None)

        result = []
        binary_targets = []
        values = []
        for parser, currency_pred, currency_nonzero, target in itertools.izip(parsers,
                                                            currency_predictions,
                                                            currency_nonzero_pred,
                                                            y):
            if not self.mapper.is_valid(parser, target):
                continue

            result.append(self.mapper.fit_transform(parser,
                                                    currency_pred,
                                                    currency_nonzero,
                                                    target))
            binary_targets.append(self.mapper.binary_targets)
            values.append(self.mapper.values)

        if len(result) > 0:
            self.binary_targets = np.concatenate(binary_targets)
            self.values = np.concatenate(values)
            return np.vstack(result)
        else:
            self.binary_targets = np.asarray([])
            self.values = np.asarray([])
            return None

    def fit_transform(self, parser, currency_predictions, currency_nonzero, y=None):
        X_transformed = self.transform(parser, currency_predictions, currency_nonzero, y)
#        k_near = 15
#        filter_KBest = SelectKBest(SCORE_FUNC, k=k_near)
    #        features_score = SCORE_FUNC(X_transformed, self.binary_targets)[0]
    #        features_score_arg = np.argsort(features_score)[::-1]
    #        print features_score
    #        print features_score_arg
    #        print np.sort(features_score)[::-1]
#        features_score_best = np.argsort(features_score)[::-1][:k_near]
#        print features_score
#        print features_score_best
#        result = filter_KBest.fit_transform(X_transformed, self.binary_targets)

#        print 'count vat features...', self.mapper.counter_vat_fetures
        result = X_transformed
        return result


class ParserTransform:
    """ Apply a parser as a transformation.

    This is a scikit-learn style transformer that assumes the input is a
    dataframe with a 'text' column and produces a list of parser objects.
    """
    def __init__(self, parser_cls):
        self.parser_cls = parser_cls

    def fit(self, X, y=None):
        pass

    def transform(self, X):
        result = [self.parser_cls(text) for text in X['text']]
        return result

    def fit_transform(self, X, y=None):
        return self.transform(X)


class ExtractionModel:
    """ Model to exact values from receipt text

    Combines a parser, feature mapper and a classifier in a single model
    that can be trained on receipt text and correct values and then allows
    prediction of receipt values on new texts.

    Parameters
    ----------
    parser : class
        The parser class used to convert the text into tokens.

    feature_mapper : class
        The class used to extract feature vectors for each token.

    classifier : sklearn style classifier
        This should be an object supporting 'fit' and 'predict_proba'
        methods.
    """
    def __init__(self,
                 currency_extractor,
                 currency_nonzero,
                 parser,
                 feature_mapper,
                 classifier,
                 selector=None):
        self.currency_extractor = currency_extractor
        self.currency_nonzero = currency_nonzero
        self.parser_transform = ParserTransform(parser)
        self.feature_transform = FeatureTransform(feature_mapper)
        self.classifier = classifier
#        self.selector = selector
#        steps = [
#            ('tokenize', ParserTransform(parser)),
#            ('feature_map', FeatureTransform(feature_mapper)),
#        ]
#        self.feature_pipeline = Pipeline(steps)

    @staticmethod
    def predict_value(subclassifier, source):
        if subclassifier is not None:
            return subclassifier.predict(source)
        else:
            return [None]*len(source)

    def _feature_pipeline_fit_transform(self, X, y=None):
        if self.currency_extractor is not None:
            currency_pred = self.currency_extractor.predict(X['text'])
        currecy_pred = self.predict_value(self.currency_extractor,
                                          X['text'])
        currency_nonzero_pred = self.predict_value(self.currency_nonzero,
                                                   X['text'])
        X_parsers = self.parser_transform.fit_transform(X, y)
        X_result = self.feature_transform.fit_transform(X_parsers,
                                                        currency_pred,
                                                        currency_nonzero_pred,
                                                        y)
        return X_result

    def predict_best(self, X):
        """ Extract values for each receipt in a sample """
        return [self._predict_single(pd.DataFrame(x).transpose())
                for (_, x) in X.iterrows()]

    def score(self, X, y):
        """ Computes average accuracy on a sample """
        predictions = self.predict_best(X)
        scores = [prediction == target
                  for (prediction, target) in zip(predictions, y)]
        return np.mean(scores)

    def fit(self, X, y):

#        Xt = self.feature_pipeline.fit_transform(X, y)
#        yt = self.feature_pipeline.steps[-1][1].binary_targets
        Xt = self._feature_pipeline_fit_transform(X, y)
        yt = self.feature_transform.binary_targets

        self.classifier.fit(Xt, yt)

    def predict(self, X):
#        Xt = self.feature_pipeline.fit_transform(X)
        Xt = self._feature_pipeline_fit_transform(X)

        if Xt is None:
            return None
        else:
            return self.classifier.predict(Xt)

    def predict_proba(self, X):
#        Xt = self.feature_pipeline.fit_transform(X)
        Xt = self._feature_pipeline_fit_transform(X)

        if Xt is None:
            return None
        else:
            return self.classifier.predict_proba(Xt)

    def _predict_single(self, X):
#        Xt = self.feature_pipeline.fit_transform(X)
        Xt = self._feature_pipeline_fit_transform(X)

        if Xt is None:
            return None
        else:
            yp = self.classifier.predict_proba(Xt)[:, 1]
            idx = np.argmax(yp)
#            return self.feature_pipeline.steps[-1][1].values[idx]
            return self.feature_transform.values[idx]


class ClassifierTester:
    def __init__(self,
                 model_list,
                 predict_flag=False,
                 proba_flag=True,
                 threshold_lst=[0.85]):
        self.model_list = model_list
        self.predict_flag = False
        self.proba_flag = True,
        self.threshold_lst = threshold_lst

    def fit(self, X, y):
        cnt_true = np.count_nonzero(y > 0.0)
        cnt_false = len(y) - cnt_true

        print 'Length of target array', len(y)
        print 'Count True values', cnt_true, float(cnt_true)/len(y)
        print 'Count False values', cnt_false, float(cnt_false)/len(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

        print "Length X_train, X_test", len(X_train), len(X_test)
        print "Length y_train, y_test", len(y_train), len(y_test)

        for model in self.model_list:
            model.fit(X_train, y_train)
            if self.predict_flag:
                self.score_model(model, X_test, y_test)
            if self.proba_flag:
                for threshold in self.threshold_lst:
                    self.score_proba(model, X_test, y_test, threshold)

    def score_model(self, model, X, y):
        yp = model.predict(X)
        print model.__class__
        print "accuracy: ", metrics.accuracy_score(y, yp)
        print "precision: ", metrics.precision_score(y, yp)
        print "recall: ", metrics.recall_score(y, yp)
        print "f1 score: ", metrics.f1_score(y, yp)

    def score_proba(self, model, X, y, threshold):

        print 'Probability threshold is: ', threshold
        y_proba = model.predict_proba(X)

        indices = np.indices(y.shape)

        loc_true = np.where(y > 0)
        loc_true = loc_true[0]
        print('Positive: %d' % len(loc_true))

        loc_false = np.where(y <= 0)
        loc_false = loc_false[0]
        print('Negative: %d' % len(loc_false))

        loc_positive = np.where(y_proba[:, [1]] > threshold)
        loc_positive = loc_positive[0]
        loc_negative = np.setxor1d(indices, loc_positive)
        print('Predicted Positive: %d' % len(loc_positive))
        print('Predicted Negative: %d' % len(loc_negative))


        loc_tp = set(loc_true) & set(loc_positive)
        loc_tp = list(loc_tp)
        print('True Positive: %d' % len(loc_tp))

        loc_fn = set(loc_true) & set(loc_negative)
        loc_fn = list(loc_fn)
#        loc_fp = loc_fp.sort()
        print('False Negative: %d' % len(loc_fn))

        loc_fp = set(loc_false) & set(loc_positive)
        loc_fp = list(loc_fp)
#        loc_fn.sort()
        print('False Positive: %d' % len(loc_fp))

#        precision = float(len(loc_tp)) / float(len(loc_tp) + len(loc_fp))
#        recall = float(len(loc_tp)) / float(len(loc_tp) + len(loc_fn))
#
#        print 'Precision calculated: ', precision
#        print 'Recall calculated: ', recall

        yp = np.zeros(y.shape, dtype=bool)
        yp[loc_positive] = True

        print "accuracy: ", metrics.accuracy_score(y, yp)
        print "precision: ", metrics.precision_score(y, yp)
        print "recall: ", metrics.recall_score(y, yp)
        print "f1 score: ", metrics.f1_score(y, yp)
