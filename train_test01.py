# -*- coding: utf-8 -*-
import data_io
from tokenization import Parser
from transforms import ExtractionModel
from total_amount_features import TotalAmountFeatures
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

NMB_ESTIMATORS = 45 # 100

feature_mappers = {
    'total_amount': TotalAmountFeatures
}

classifiers = {
    'total_amount': RandomForestClassifier(n_estimators=NMB_ESTIMATORS,
                                           random_state=1)
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


def extract_directory(source_dir, destination_dir, indexes, source_data):

    os.mkdir(destination_dir)

    destination_data = source_data.loc[indexes]

    header_columns = ['id']+destination_data.columns.tolist()

    destination_data.to_csv(destination_dir+'data.csv',
                            index_label='id')

    for idx in indexes:
        fname = str(idx)+'.txt'
        shutil.copyfile(source_dir + fname,
                        destination_dir + fname)


##############

##############
#dir_name = '/home/evgeniy/ReceiptBank/rb-engine/text-training-old/'
# match_file = 'data.csv'
dir_name = '/home/evgeniy/ReceiptBank/rb-engine/text-training/'
match_file = 'data.csv'


#==============================================================================
#path = dir_name + match_file
# 
#data00 = pd.read_csv(path, index_col='id')
# 
#data01 = pd.read_csv(path)
# 
#X_train, X_test, y_train, y_test = \
#    cross_validation.train_test_split(data01,
#                                      data01['total_amount'],
#                                      test_size=0.25)
# 
# 
#train_indexes = [it[0] for it in X_train ]
#test_indexes = [it[0] for it in X_test ]
# 
#extract_directory(dir_name,
#                  '/home/evgeniy/ReceiptBank/rb-engine/text-training-old/',
#                  train_indexes,
#                  data00
#                  )
# 
#extract_directory(dir_name,
#                  '/home/evgeniy/ReceiptBank/rb-engine/text-test-old/',
#                  test_indexes,
#                  data00
#                  )
#==============================================================================

data = data_io.load_from_files(dir_name)
fields = ['total_amount']

print 'before training...........'

models = {}

for field in fields:
    models[field] = train_model(data, field)

with open('models_est45.p', 'w') as model_file:
    pickle.dump(models, model_file)
############
