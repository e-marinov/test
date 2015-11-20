# -*- coding: utf-8 -*-
"""
Created

@author: evgeniy
"""
import data_io
from tokenization import Parser
from transforms import ExtractionModel
from total_amount_features import TotalAmountFeatures
from tax_amount_features import TaxAmountFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.cross_validation import cross_val_score
from sklearn import cross_validation

import score_model

import numpy as np

import sys
import os
import shutil
import decimal
import pickle
import pandas as pd

NMB_ESTIMATORS = 35 # 100

feature_mappers = {
    'total_amount': TotalAmountFeatures,
    'vat_amount': TaxAmountFeatures
}

classifiers = {
    'total_amount': RandomForestClassifier(n_estimators=NMB_ESTIMATORS,
                                           random_state=1),
    'vat_amount': RandomForestClassifier(n_estimators=NMB_ESTIMATORS,
                                           random_state=1),
                                           
}

feature_selectors_cl = {
    'total_amount': SelectKBest(f_classif, k=10)
}

feature_selectors_chi2 = {
    'total_amount': SelectKBest(chi2, k=10)
}


def train_model(data, var_name):
    targets = [decimal.Decimal("%.2f" % v) for v in data[var_name]]
    model = ExtractionModel(Parser,
                            feature_mappers[var_name],
                            classifiers[var_name])
    model.fit(data, targets)
    return model


##############
#dir_name = '/home/evgeniy/ReceiptBank/rb-engine/text-training-old/'
# match_file = 'data.csv'

dir_train = '/home/evgeniy/ReceiptBank/rb-engine/text-training-small/'
dir_test = '/home/evgeniy/ReceiptBank/rb-engine/text-training-small/'

#dir_train = '/home/evgeniy/ReceiptBank/rb-engine/text-training/'
#dir_test = '/home/evgeniy/ReceiptBank/rb-engine/text-test/'
match_file = 'data.csv'
##############


print 'Before training...........'

data = data_io.load_from_files(dir_train)

# model = train_model(data, 'total_amount')
amount_name = 'total_amount'

model = train_model(data, amount_name)


print 'After training...........'


data = data_io.load_from_files(dir_test)

res = model.score(data, data[amount_name])

print "Score: %.4f" % res

############
