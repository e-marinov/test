import numpy as np
import operator as op
import pickle
from decimal import Decimal

epsilon = Decimal('0.015')

vat_rates = {'GBP': (20, ['GBP', 'GB', 'GBR', 'UK', 'KINGDOM', 'BRITAIN',
                          'LONDON']),
             'AUD': (10, ['AUD', 'AU', 'AUS', 'AUSTRALIA', 'SYDNEY',
                          'MELBOURNE', 'BRISBANE']),
             'NZD': (15, ['NZD', 'NZ', 'NZL', 'ZEALAND']),
             'CAD': (5, ['CAD', 'CA', 'CAN', 'CANADA', 'ONTARIO', 'MONTREAL'])}


class TotalAmountFeatures:
    """ Feature mapper for the tax amount field. """
    def __init__(self):
#        with open('/home/evgeniy/ReceiptBank/rb-engine/currency_model.p') \
#                                                          as model_file:
#            self.currency_model = pickle.load(model_file)
        print '..intotal amount features...'
#
        self.doc_counter = 0

    def fit(self, parser, currency_pred, currency_nonzero):
        numeric_tokens = parser.numeric_tokens
        self.values = [token.value for token in numeric_tokens]
        self.maximum = max(self.values)
        self.minimum = max(self.values)
        self.mean = np.mean(self.values)
        self.median = np.median(self.values)

        self.predicted_currency = currency_pred
        self.currency_nonzero = currency_nonzero

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

    def fit_transform(self, parser, currency_pred, currency_nonzero, y=None):
        self.doc_counter += 1

        self.fit(parser, currency_pred, currency_nonzero)
        return self.transform(parser, y)

    def is_valid(self, parser, target=None):
        return len(parser.numeric_tokens) > 0

    ###################################################
    def _feature_VAT(self, total, currency, vat_rate, filter_op):

        if self.predicted_currency != currency: # or self.currency_nonzero is False:
            return self._encode_binary(False)

        def verify_total_vat(total, tax, vat_rate):
            return abs((tax*100)/(total - tax) - vat_rate) < epsilon

        exists_vat = any(verify_total_vat(total, val, vat_rate)
                          for val in self.values
                          if filter_op(val, total) and float(val) != 0.0)

        return self._encode_binary(exists_vat)

    def _vat_feature_vector(self, total, filter_op):
        feature_holder = []
        for currency, rates in vat_rates.iteritems():
            tmp_values = [self._feature_VAT(total,
                                            currency,
                                            vat_rate,
                                            filter_op) for vat_rate in rates]
            feature_holder.append(tmp_values)
        return np.asarray(feature_holder)

    def _feature_vector(self, token):

        return np.asarray([
            self._encode_binary(token.value == self.maximum),           # 0
            self._encode_binary(token.value >=                          # 1
                                sum((x for x in self.values
                                     if x < token.value))),
            self._encode_binary(token.value >= self.mean),              # 2
            self._encode_binary(token.value >= self.median),            # 3
            self._encode_binary(round(token.value) == token.value),     # 4
            self._is_at_index(-1, token.value),                         # 5
            self._is_at_index(-2, token.value),                         # 6
            self._contains(token.line_prefix, 'total'),                 # 7
            self._contains(token.previous_line, 'total'),               # 8
#            self._contains(token.line_prefix, 'change'),                # --9
#            self._contains(token.previous_line, 'change'),              # --10
            self._contains(token.line_prefix, 'tax'),                   # 9
            self._contains(token.previous_line, 'tax'),                 # 10
            self._contains(token.line_prefix, 'vat'),                   # 11
            self._contains(token.previous_line, 'vat'),                 # 12
            self._contains(token.line_prefix, 'gst'),                   # 13
            self._contains(token.previous_line, 'gst'),                 # 14
            self._contains(token.string, '.'),                          # 15
            self._contains(token.string, ','),                          # 16
            self._is_sequence_present([token.value, self.maximum,       # 17
                                       self.maximum - token.value]),
            self._is_sequence_present([self.maximum, self.maximum - token.value,
                                       token.value]),                   # 18
            self._is_sequence_present([self.maximum, token.value,       # 19
                                       self.maximum - token.value]),
            self._is_sequence_present([self.maximum, token.value]),     # 20
            self._is_sequence_present([self.maximum, token.value], 2),  # 21
            self._is_sequence_present([token.value, self.maximum]),     # 22
            self._is_sequence_present([token.value, self.maximum], 2),  # 23
            self._is_sequence_present([token.value, token.value]),      # 24

            self._feature_VAT(token.value, 'GBP', 20, op.lt),              # 26
            self._feature_VAT(token.value, 'AUD', 10, op.lt),              # 27
            self._feature_VAT(token.value, 'NZD', 15, op.lt),              # 28
            self._feature_VAT(token.value, 'CAD', 13, op.lt),              # 29
            self._feature_VAT(token.value, 'CAD', 5,  op.lt),              # 30
            self._feature_VAT(token.value, 'EUR', 21, op.lt),              # 31
            self._feature_VAT(token.value, 'EUR', 23, op.lt),              # 32
#

#            self._feature_VAT(token.value, 'GBP', Decimal('20.0'), op.lt),        # 25
##            self._feature_VAT(token, 'GBP', op.ne),                      # ----26
#            self._feature_VAT(token.value, 'AUD', Decimal('10.0'), op.lt),        # 26
##            self._feature_VAT(token, 'AUD', op.ne),                      # --- 28
#            self._feature_VAT(token.value, 'CAD', Decimal('5.0'), op.lt),                      # 27
##            self._feature_VAT(token, 'CAD', op.ne, use_signs=True),      # --- 30
#            self._feature_VAT(token.value, 'NZD', Decimal('15.0'), op.lt),                      # 28
##            self._feature_VAT(token, 'NZD', op.ne),                      # --- 32
        ])

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
