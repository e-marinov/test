from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model.stochastic_gradient import SGDClassifier
from sklearn.multiclass import OneVsRestClassifier

from scipy.stats import uniform as sp_rand
from sklearn.grid_search import RandomizedSearchCV
from sklearn import preprocessing

from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.datasets import fetch_20newsgroups
from stop_words import get_stop_words
import numpy as np
import pandas as pd
import pickle
import os

stp_words = get_stop_words('en')


class CurrencyClassifier(object):

    def __init__(self, base_estimator, currency_codes):
        self.currency_codes = currency_codes

        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(min_df=2, # 0.9478
#                                      use_idf=False,
                                      norm=None,
                                      ngram_range=(1, 4),
                                      stop_words=stp_words)
                                      ),
            ('clf', base_estimator)])
#
#            ('clf', OneVsRestClassifier(estimator=base_estimator,
#                                        multilabel_=False))])

    @classmethod
    def from_pickle(cls, file_path):
        with open(file_path) as classifier_file:
            return pickle.load(classifier_file)

#    def score_currencies(Xtest, ytest):
#        scores = {}
#        Xtest[Xtest['currency_code']==curr]['msgs']

    def fit(self, Xtrain, ytrain):
        y_transformed = map(self.encode, ytrain)
        return self.pipeline.fit(Xtrain, y_transformed)

    def score(self, Xtest, ytest):
        y_test = map(self.encode, ytest)
        return self.pipeline.score(Xtest, y_test)

    def predict(self, Xtest):
        y_predicted = self.pipeline.predict(Xtest)
        return map(self.decode, y_predicted)

    def predict_signle(self, msg):
        return self.predict([msg])[0]

    def save_classifier(self, file_name):
        with open(file_name, 'w') as model_file:
            pickle.dump(self, model_file)

    def encode(self, curr):
        if curr in self.currency_codes:
            return self.currency_codes.index(curr)
        else:
            return len(currency_cod) - 1

    def decode(self, cod):
        return self.currency_codes[cod]

############################
path_docfiles = '/home/evgeniy/ReceiptBank/rb-engine/rb_engine_dataset'

currency_cod = ['GBP', 'AUD', 'NZD', 'CAD', 'EUR', 'USD', 'OTHER']


def get_doc_msg(doc_id):
    with open(os.path.join(path_docfiles, str(doc_id) + '.txt')) as foo:
        lines = foo.readlines()
        msg = ''.join(lines)
    return msg


def load_top_currencies(data, other_curr):
    indexes = data[~data.currency_code.isin(currency_cod)].index
    data.loc[indexes, 'currency_code'] = other_curr
    data['text'] = map(get_doc_msg, data.index)
    return data


def verify_total_vat(total, tax, vat_rate):
    return abs((tax*100)/(total - tax) - vat_rate) < epsilon


def vat_rate(total, tax):
    return (tax*100)/(total - tax)


def curr_stats(data, curr):
    epsilon = 0.05

    def check_rate(val, rate):
        if abs(val - rate) < epsilon:
            return True
        else:
            return False

    rates = range(1, 50, 1)
    res = {}
    for c in curr:
        stat_holder = {}
        data_c = data[data.currency_code == c]
        nultax_mask = data_c.vat_amount == 0.0
        stat_holder[0] = np.count_nonzero(nultax_mask)
        data_nonzero_tax = data_c[~nultax_mask]

        for r in rates:
            lst = map(vat_rate,
                      data_nonzero_tax.total_amount,
                      data_nonzero_tax.vat_amount)
            count = np.count_nonzero(map(check_rate, lst, [r]*len(lst)))
            if count > 250:  # or count > len(data_nonzero_tax) / 20.0:
                stat_holder[r] = count
        res[c] = stat_holder
    return res


def main():

    print 'Inside script...'

#    data = pd.read_csv('/home/evgeniy/ReceiptBank/rb-engine/csvs/user_receipts.csv')

    columns_needed = ['currency_code']
    data_train = pd.read_csv('/home/evgeniy/ReceiptBank/rb-engine/rb_digital_split/train_large120/data.csv',
                             index_col='id')
    data_train = data_train[columns_needed]

    data_test = pd.read_csv('/home/evgeniy/ReceiptBank/rb-engine/rb_digital_split/test_large30/data.csv',
                            index_col='id')

    data_test = data_test[columns_needed]

    data_train = load_top_currencies(data_train, currency_cod[-1])
    data_test = load_top_currencies(data_test, 'OTHERR1')

    #twenty = fetch_20newsgroups()
    #
    #tfidf = TfidfVectorizer().fit_transform(twenty.data)
    #tfidf

    estimator = LinearSVC(
        class_weight='auto',
        random_state=1)

    classifier = CurrencyClassifier(estimator, currency_cod)

    classifier.fit(data_train.text, data_train.currency_code)

    # NMB_ESTIMATORS = 35

    #estimator = RandomForestClassifier( # n_estimators=NMB_ESTIMATORS,
    #                                  random_state=1, class_weight='auto')
    #estimator = SGDClassifier(
    #    random_state=1,
    #    class_weight='auto',
    #    loss='hinge',
    #    alpha=0.00001,
    #    penalty='elasticnet'
    #    )

#    scores = classifier.pipeline.score(data_test.text, data_test.currency_code)
    pred_curr = classifier.predict(data_test.text)
    res = data_test.currency_code == pred_curr
    print len(res.nonzero()[0])/float(len(res))

#    dt_test = data_test[data_test.currency_code != 'OTHER']
#    pred_test = pred_curr[np.where(data_test.currency_code != 'OTHER')[0]]
#    res_test = dt_test.currency_code == pred_test
#    print len(res_test.nonzero()[0])/float(len(res_test))

    classifier.save_classifier('currency_model_stop_w.p')


    #start = timer()
    #
    #
    #end = timer()
    #print end - start

if __name__ == "__main__":
    main()
