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

import numpy as np
import pandas as pd
import pickle
import os


class CurrencyZeroClassifier:

    def __init__(self, base_estimator):
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(min_df=2, ngram_range=(1, 3))),
            ('clf', base_estimator)])

    @classmethod
    def from_pickle(cls, file_path):
        with open(file_path) as classifier_file:
            return pickle.load(classifier_file)

    def fit(self, Xtrain, ytrain):
        y_train = map(self.encode, ytrain)
        return self.pipeline.fit(Xtrain, y_train)

    def predict(self, Xtest):
        predicted = self.pipeline.predict(Xtest)
        return map(self.decode, predicted)

    def predict_signle(self, msg):
        return self.predict([msg])[0]

    def score(self, Xtest, ytest):
        y_test = map(self.encode, ytest)
        return self.score(Xtest, y_test)

    def save_classifier(self, file_name):
        with open(file_name, 'w') as model_file:
            pickle.dump(self, model_file)

    def encode(self, val):
        if val == 0.0:
            return 0.0
        else:
            return 1.0

    def decode(self, cod):
        if cod == 0.0:
            return False
        else:
            return True


############################
path_docfiles = '/home/evgeniy/ReceiptBank/rb-engine/rb_engine_dataset'

currency_cod = ['GBP', 'AUD', 'NZD', 'CAD', 'EUR', 'USD', 'OTHER']


def get_doc_msg(doc_id):
    with open(os.path.join(path_docfiles, str(doc_id) + '.txt')) as foo:
        lines = foo.readlines()
        msg = ''.join(lines)
    return msg


def main():

    print 'Inside script...'

#    data = pd.read_csv('/home/evgeniy/ReceiptBank/rb-engine/csvs/user_receipts.csv')

    columns_needed = ['currency_code', 'vat_amount']
    data_train = pd.read_csv('/home/evgeniy/ReceiptBank/rb-engine/rb_digital_split/train_large120/data.csv',
                             index_col='id')
    data_train = data_train[columns_needed]

    data_test = pd.read_csv('/home/evgeniy/ReceiptBank/rb-engine/rb_digital_split/test_large30/data.csv',
                            index_col='id')
    data_test = data_test[columns_needed]

    data_train['text'] = map(get_doc_msg, data_train.index)
    data_test['text'] = map(get_doc_msg, data_test.index)

    data_train['target'] = map(float, data_train['vat_amount'] == 0.0)
    data_test['target'] = map(float, data_test['vat_amount'] == 0.0)

#    tfidf_vectorizer = TfidfVectorizer(min_df=2, ngram_range=(1, 3))
    
    #twenty = fetch_20newsgroups()
    #
    #tfidf = TfidfVectorizer().fit_transform(twenty.data)
    #tfidf
    estimator = LinearSVC(
        class_weight='auto',
        random_state=1)


#    estimator = SGDClassifier(
#        random_state=1,
#        class_weight='auto',
##        loss='hinge',
##        alpha=0.00001,
##        penalty='elasticnet'
#        )

    classifier = CurrencyZeroClassifier(estimator)
    classifier.fit(data_train.text, data_train.vat_amount)

#    classifier = Pipeline([
#    #    ('vectorizer', CountVectorizer()),
#        ('tfidf', TfidfVectorizer(min_df=2, max_df=0.2, ngram_range=(1,3))),
#        ('clf', estimator)])

    predicted = classifier.predict(data_test.text)

    res = predicted == (data_test.vat_amount != 0.0)
    print 'accuracy: ', float(np.count_nonzero(res))/len(res)

    scores = classifier.score(data_test.text, data_test.vat_amount)
    print scores
    classifier.save_classifier('currency_zero_classifier.p')
  
   
    stop = a
if __name__ == "__main__":
    main()
