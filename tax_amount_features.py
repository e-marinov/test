import numpy as np
import operator as op
import pickle

epsilon = 0.15

currency_to_sign = {'GBP': ['GBP', 'GB', 'GBR', 'UK', 'Kingdom'],
                    'AUD': ['AUD', 'AU', 'AUS', 'AUSTRALIA'],
                    'NZD': ['NZD', 'NZ', 'NZL', 'ZEALAND'],
                    'CAD': ['CAD', 'CA', 'CAN', 'CANADA']}

currency_to_name = {'GBP': [('United', 'Kingdom'),
                            ('BRITAIN'),
                            ('GREAT', 'BRITAIN'),
                            ('London')],
                    'AUD': ['AUD', 'AU', 'AUS', 'AUSTRALIA', 'GREAT BRITAIN'],
                    'NZD': ['NZD', 'NZ', 'NZL', 'ZEALAND'],
                    'CAD': ['CAD', 'CA', 'CAN', 'CANADA']}


vat_rates = {'GBP': [20], #  , 5],
             'AUD': [10],
             'NZD': [15],
             'CAD': [13, 5],
             'EUR': [21, 23]}


class TaxAmountFeatures:
    """ Feature mapper for the tax amount field. """
    def __init__(self):
        print '..in tax amount features..'
#        with open('/home/evgeniy/ReceiptBank/rb-engine/currency_model.p') \
#                                                          as model_file:
#            self.currency_model = pickle.load(model_file)
#
        self.doc_counter = 0
        self.counter_vat_fetures = 0

    def fit(self, parser, currency_pred, currency_nonzero=None):
        numeric_tokens = parser.numeric_tokens
        self.values = [token.value for token in numeric_tokens]
        self.maximum = max(self.values)
        self.minimum = max(self.values)
        self.mean = np.mean(self.values)
        self.median = np.median(self.values)

        self.predicted_currency = currency_pred
#        self.currency_nonzero = currency_nonzero

    def transform(self, X, y=None):
        if y is None:
            self.binary_targets = [None]*len(X.numeric_tokens)
        else:
            binary_true = self._encode_binary(True)
            binary_false = self._encode_binary(False)
            self.binary_targets = np.where(np.asarray(self.values) == y,
                                           binary_true,
                                           binary_false)

        return np.vstack([self._feature_vector(token)
                          for token in X.numeric_tokens])

    def fit_transform(self,
                      parser,
                      currency_pred,
                      currency_nonzero=None,
                      y=None):
        self.doc_counter += 1

        self.fit(parser, currency_pred, currency_nonzero=None)
        return self.transform(parser, y)

    def is_valid(self, parser, target=None):
        return len(parser.numeric_tokens) > 0

    def _exists_sign_intoken(self, token, currency):
        vat_rate, signs = vat_rates[currency]
        exists_sign = any(token.source.contain_word(sign)
                          for sign in signs)
        return exists_sign

    def _exists_usd_intoken(self, token):
        usd_strings = ['USD', 'USA', 'AMERICA']
        usd_strings_exact = ['U.S', 'UNITED STATES']
        exists_name = any(token.source.contain_word(val)
                          for val in usd_strings)
        exists_name_exact = any(token.source.contain_word_exact(val)
                                for val in usd_strings_exact)
        return (exists_name or exists_name_exact)

    def _feature_VAT(self, tax, currency, vat_rate, filter_op):

        if self.predicted_currency != currency: # or self.currency_nonzero is False:
            return self._encode_binary(False)

        def verify_total_vat(total_, tax_, vat_rate_):
            return abs((tax_*100)/(total_ - tax_) - vat_rate_) < epsilon

        exists_vat = any(verify_total_vat(val, tax, vat_rate)
                         for val in self.values
                         if filter_op(val, tax) and float(val) != 0.0)

        return self._encode_binary(exists_vat)

    def _vat_feature_vector(self, tax, filter_op):
        feature_holder = []
        for currency, rates in vat_rates.iteritems():
            tmp_values = [self._feature_VAT(tax,
                                            currency,
                                            vat_rate,
                                            filter_op) for vat_rate in rates]
            feature_holder += tmp_values
        return np.asarray(feature_holder)

#
#    def _feature_VAT(self, token, currency, filter_op):
#        vat_rates_c = vat_rates[currency]
#
#        if self.predicted_currency != currency:
#            return self._encode_binary(False)
#
#        def verify_total_vat(total, tax, vat_rate):
#            return abs((tax*100)/(total - tax) - vat_rate) < epsilon
#
#        def check_rate(vat_rate):
#            return any(verify_total_vat(val, token.value, vat_rate)
#                       for val in self.values
#                       if filter_op(val, token.value) and float(val) != 0.0)
#
#        exists_vat = any(check_rate(rate) for rate in vat_rates_c)
#
#        return self._encode_binary(exists_vat)

    def _feature_vector(self, token):

        features = np.asarray([
            self._encode_binary(token.value == 0.0),
            self._encode_binary(self.predicted_currency == 'USD'),
            self._encode_binary(token.value < self.maximum),
            self._encode_binary(token.value >=
                                sum((x for x in self.values
                                     if x < token.value))),
            self._encode_binary(token.value >= np.mean(self.values)),      # 4
            self._encode_binary(token.value >= np.median(self.values)),    # 5
            self._encode_binary(round(token.value) == token.value),        # 6 
            self._is_at_index(-1, token.value),                            # 7
            self._is_at_index(-2, token.value),                            # 8
#            self._contains(token.line_prefix, 'total'),
            self._contains(token.previous_line, 'total'),                  # 9 
#            self._contains(token.line_prefix, 'change'),
#            self._contains(token.previous_line, 'change'),
            self._contains(token.line_prefix, 'tax'),                   # 10
            self._contains(token.previous_line, 'tax'),                 # 11
            self._contains(token.line_prefix, 'vat'),                   # 12
            self._contains(token.previous_line, 'vat'),                 # 13
            self._contains(token.line_prefix, 'gst'),                   # 14
            self._contains(token.previous_line, 'gst'),                    # 15 ------
            self._contains(token.string, '.'),                             # 16 ------
            self._contains(token.string, ','),                             # 17
            self._is_sequence_present([token.value, self.maximum,          # 18 
                                       self.maximum - token.value]),
            self._is_sequence_present([self.maximum, self.maximum - token.value,
                                       token.value]),                      # 19
            self._is_sequence_present([self.maximum, token.value,          # 20 
                                       self.maximum - token.value]),
            self._is_sequence_present([self.maximum, token.value]),        # 21
            self._is_sequence_present([self.maximum, token.value], 2),     # 22
            self._is_sequence_present([token.value, self.maximum]),        # 23
            self._is_sequence_present([token.value, self.maximum], 2),     # 24
            self._is_sequence_present([token.value, token.value]),         # 25
            self._feature_VAT(token.value, 'GBP', 20, op.gt),              # 26
            self._feature_VAT(token.value, 'AUD', 10, op.gt),              # 27
            self._feature_VAT(token.value, 'NZD', 15, op.gt),              # 28
            self._feature_VAT(token.value, 'CAD', 13, op.gt),              # 29
            self._feature_VAT(token.value, 'CAD', 5,  op.gt),              # 30
            self._feature_VAT(token.value, 'EUR', 21, op.gt),              # 31
            self._feature_VAT(token.value, 'EUR', 23, op.gt),              # 32
        ])

#        currency_features = self._vat_feature_vector(token.value, op.gt)
        if features[-7:].sum() > 0.0:
            self.counter_vat_fetures += 1
#        return np.hstack((features, currency_features))
        return features

    def _encode_binary(self, value):
        if value:
            return 1.0
        else:
            return 0.0

    def _contains(self, string, word):
        return self._encode_binary(string.lower().find(word) >= 0)

    def _is_at_index(self, idx, v):
        try:
            ref = self.values[idx]
        except:
            ref = None
        return self._encode_binary(ref == v)

    def _is_sequence_present(self, sequence, spacing=1):
        size = (len(sequence) - 1)*spacing + 1
        if len(self.values) < size:
            return self._encode_binary(False)
        lst = [self.values[i:i + size:spacing]
               for i in range(len(self.values) - len(sequence))]
        return self._encode_binary(sequence in lst)
