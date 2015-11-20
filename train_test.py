import data_io
from tokenization import Parser
from transforms import ExtractionModel
from total_amount_features import TotalAmountFeatures
from tax_amount_features import TaxAmountFeatures
from currency_tfidf_script import CurrencyClassifier
from currency_zero_tax_script import CurrencyZeroClassifier

import sys
sys.path.append('/home/evgeniy/.local/lib/python2.7/site-packages ')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.cross_validation import cross_val_score
from sklearn import cross_validation


import score_model

import numpy as np

import os
import shutil
import decimal
import pickle
import pandas as pd


NMB_ESTIMATORS = 35  # 100

feature_mappers = {
    'total_amount': TotalAmountFeatures,
    'vat_amount': TaxAmountFeatures
}

classifiers = {
    'total_amount': RandomForestClassifier(n_estimators=NMB_ESTIMATORS,
                                           random_state=1, class_weight='auto'),
    'vat_amount': RandomForestClassifier(n_estimators=NMB_ESTIMATORS,
                                         random_state=1),
}

feature_selectors_cl = {
    'total_amount': SelectKBest(f_classif, k=10)
}

feature_selectors_chi2 = {
    'total_amount': SelectKBest(chi2, k=10)
}


def train_model(data, var_name, currency_extractor, currency_nonzero):
    targets = [decimal.Decimal("%.2f" % v) for v in data[var_name]]
    model = ExtractionModel(currency_extractor,
                            currency_nonzero,
                            Parser,
                            feature_mappers[var_name],
                            classifiers[var_name])
    model.fit(data, targets)
    return model


def test_vat_retrieval():

    #dir_test = '/home/evgeniy/ReceiptBank/rb-engine/rb_digital_split/test4/'
    #dir_train = '/home/evgeniy/ReceiptBank/rb-engine/rb_digital_split/train30/'
    
#    dir_test = '/home/evgeniy/ReceiptBank/rb-engine/text-training-small'
#    dir_train = dir_test

    dir_test = '/home/evgeniy/ReceiptBank/rb-engine/rb_digital_split/test_large30/'
    dir_train = '/home/evgeniy/ReceiptBank/rb-engine/rb_digital_split/train_large120/'

    #amount_name = 'vat_amount'
    amount_name = 'vat_amount'

    #dir_train = '/home/evgeniy/ReceiptBank/rb-engine/text-training/'
    #dir_test = '/home/evgeniy/ReceiptBank/rb-engine/text-test/'
    ##############
    print 'READING............'

    data = data_io.load_from_files(dir_train)

    print 'TRAINING...........'

    # model = train_model(data, 'total_amount')
    # amount_name = 'total_amount'
    print amount_name

    data_nonzerovat = data[data['vat_amount'] != 0.0]

    print 'loading currency extractor...'
    currency_estractor = CurrencyClassifier.from_pickle('currency_model_.p')

    model = train_model(data_nonzerovat,
                        amount_name,
                        currency_estractor,
                        currency_nonzero=None)

    print 'TESTING...........'

    data = data_io.load_from_files(dir_test)

    print 'loading currency zero classifier....'

    currency_nonzero = CurrencyZeroClassifier.from_pickle('currency_zero_classifier.p')

    predict_nonzero = currency_nonzero.predict(data['text'])
    predict_zero = map(lambda x: not x, predict_nonzero)

    data_zero_vat = data[predict_zero]
    zero_nmb = len(data_zero_vat[data_zero_vat.vat_amount == 0.0])
    zero_vat_accuracy = zero_nmb / float(len(data_zero_vat))

    print "Score (from zero): %.4f" % zero_vat_accuracy
    print "Classified from zero data: %.4f" % (float(len(data_zero_vat))/len(data))
    real_score_zero = zero_vat_accuracy * (float(len(data_zero_vat))/len(data))
    print "Real score: %.4f" % real_score_zero

    data_nonzero_predicted = data[predict_nonzero]
    res = model.score(data_nonzero_predicted, data_nonzero_predicted[amount_name])

    print "==========================="
    print "Score (from nonzero): %.4f" % res
    print "Classified from nonzero data: %.4f" % (float(len(data_nonzero_predicted))/len(data))
    real_score_nonzero = res * float(len(data_nonzero_predicted))/len(data)
    print "Real score: %.4f" % real_score_nonzero
    ############


def test_total_retrieval():

    dir_test = '/home/evgeniy/ReceiptBank/rb-engine/rb_digital_split/test_large30/'
    dir_train = '/home/evgeniy/ReceiptBank/rb-engine/rb_digital_split/train_large120/'

#    dir_test = '/home/evgeniy/ReceiptBank/rb-engine/text-training-small'
#    dir_train = dir_test
    amount_name = 'total_amount'

    print 'READING............'

    data = data_io.load_from_files(dir_train)
#    data = data[data.vat_amount == 0.0]

    data_test = data_io.load_from_files(dir_test)
#    data_test = data_test[data_test.vat_amount == 0.0]
    print 'len of testing data...', len(data_test)


    print 'TRAINING...........'
    print 'len of training data...', len(data)

    # model = train_model(data, 'total_amount')
    # amount_name = 'total_amount'
    print amount_name

#    data_nonzerovat = data[data['vat_amount'] != 0.0]

    print 'loading currency extractor...'
    currency_estractor = CurrencyClassifier.from_pickle('currency_model_.p')
    currency_nonzero = CurrencyZeroClassifier.from_pickle('currency_zero_classifier.p')

    model = train_model(data,
                        amount_name,
                        currency_estractor,
                        currency_nonzero)

    print 'TESTING...........'

    res = model.score(data_test, data_test[amount_name])
    print 'score: ', res


def main():

    test_total_retrieval()
#    test_vat_retrieval()
    
if __name__ == "__main__":
    main()
