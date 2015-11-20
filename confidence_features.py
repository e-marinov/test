import numpy as np
import pandas as pd

import dateutil
import datetime


MISSING_ID = -1
MISSING_DATE = 0
SECONDS_PER_DAY = 24*60*60


def compute_statistics(data, columns):
    counts = data['created_at'].count()
    result = pd.DataFrame(index=counts.index)
    result['sample_size'] = counts

    for name in columns:
        result[name + '_min'] = data[name].min()
        result[name + '_avg'] = data[name].mean()
        result[name + '_max'] = data[name].max()

    return result


class ConfidenceFeatures:
    def __init__(self, X):
        item_age = (X['created_at'] - X['date'])/SECONDS_PER_DAY
        X['item_age'] = np.where(X['date'] != MISSING_DATE,
                                 item_age,
                                 MISSING_DATE)

        self.stats_by_user = compute_statistics(
            X.groupby(['user_id']),
            ['total_amount', 'item_age']
        )
        self.stats_by_user_shop = compute_statistics(
            X.groupby(['user_id', 'shop_id']),
            ['total_amount', 'item_age']
        )
        self.stats_by_user_currency = compute_statistics(
            X.groupby(['user_id', 'currency_id']),
            ['total_amount']
        )

    def fit(self, X, y=None):
        pass

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def transform(self, X):
        keys = ['id'] + list(X.columns)
        Xt = np.vstack(
            [self._feature_vector(dict(zip(keys, t))) for t in X.itertuples()]
        )

#        np.savetxt("features_values01.csv", Xt, delimiter=",")

        return Xt

    def _encode_binary(self, value):
        if value:
            return 1.0
        else:
            return -1.0

    def _feature_user_has_supplier(self, user_id, shop_id):
        """ Checks if the user already has the supplier.
        """
        rst = ((user_id, shop_id) in self.stats_by_user_shop.index)
        return self._encode_binary(rst)

    def _feature_user_has_currency(self, user_id, currency_id):
        """ Checks if the user already has items with this currency.
        """
        rst = ((user_id, currency_id) in self.stats_by_user_currency.index)
        return self._encode_binary(rst)

    def _feature_avg_total_range(self, target_total, user_id):
        """ Checks if the average total of the user items
        is within a range (for ex. 0.5 < total / avg_total < 1.5).
        """
        if user_id in self.stats_by_user.index:
            avg_total = self.stats_by_user.ix[user_id, 'total_amount_avg']
            total_frac = target_total/avg_total
            return self._encode_binary((0.5 < total_frac) & (total_frac < 1.5))
        else:
            return self._encode_binary(False)

    def _feature_avg_total_range_shop(self, target_total, user_id, shop_id):
        """ Checks if the average total of the user items by shop
        is within a range (for ex. 0.5 < total / avg_total < 1.5).
        """
        if (user_id, shop_id) in self.stats_by_user_shop.index:
            avg_total_by_shop = self.stats_by_user_shop.ix[(user_id, shop_id), 'total_amount_avg']
            total_frac = target_total/avg_total_by_shop
            return self._encode_binary((0.5 < total_frac) & (total_frac < 1.5))
        else:
            return self._encode_binary(False)
#################

    def _feature_total_minmax(self, target_total, user_id):
        """ Checks if the total of the item
        is greater than the minimal total for this user and
        less than the maximal total for this user, i.e.
        min_total < total < max_total.
        """
        if user_id in self.stats_by_user.index:
            min_total = self.stats_by_user.ix[user_id, 'total_amount_min']
            max_total = self.stats_by_user.ix[user_id, 'total_amount_max']
            rst = (min_total < target_total) & (target_total < max_total)
            return self._encode_binary(rst)
        else:
            return self._encode_binary(False)

    def _feature_total_minmax_shop(self, target_total, user_id, shop_id):
        """ Checks if the total of the item
        is greater than the minimal total for this user by shop and
        less than the maximal total for this user by shop, i.e.
        min_total < total < max_total.
        """
        if (user_id, shop_id) in self.stats_by_user_shop.index:
            min_total = self.stats_by_user_shop.ix[(user_id, shop_id), 'total_amount_min']
            max_total = self.stats_by_user_shop.ix[(user_id, shop_id), 'total_amount_max']
            rst = (min_total < target_total) & (target_total < max_total)
            return self._encode_binary(rst)
        else:
            return self._encode_binary(False)

    ###############

    def _feature_avg_item_age(self, target_item_age, user_id):
        """ Checks if the average age of the user items
        is within a range (for ex. 0.5 < total / avg_total < 1.5).
        """
        if user_id in self.stats_by_user.index:
            avg_age = self.stats_by_user.ix[user_id, 'item_age_avg']
            if np.isnan(avg_age):
                avg_age = 0.0
            if abs(avg_age) < 0.001:
                return self._encode_binary(False)

            age_frac = target_item_age/avg_age
            return self._encode_binary((0.5 < age_frac) & (age_frac < 1.5))
        else:
            return self._encode_binary(False)

    def _feature_avg_item_age_shop(self, target_item_age, user_id, shop_id):
        """ Checks if the average age of the user items by shop
        is within a range (for ex. 0.5 < total / avg_total < 1.5).
        """
        if (user_id, shop_id) in self.stats_by_user_shop.index:
            avg_age = self.stats_by_user_shop.ix[(user_id, shop_id), 'item_age_avg']
            if np.isnan(avg_age):
                avg_age = 0.0
            if abs(avg_age) < 0.001:
                return self._encode_binary(False)

            age_frac = target_item_age/avg_age
            rst = (0.5 < age_frac) & (age_frac < 1.5)
            return self._encode_binary(rst)
        else:
            return self._encode_binary(False)

    #################

    def _feature_minmax_item_age(self, target_age, user_id):
        """ Checks if the age of the item
        is greater than the minimal total for this user and
        less than the maximal age for this user, i.e.
        min_age < age < max_age.
        """
        if user_id in self.stats_by_user.index:
            min_age = self.stats_by_user.ix[user_id, 'item_age_min']
            max_age = self.stats_by_user.ix[user_id, 'item_age_max']

            rst = (min_age < target_age) & (target_age < max_age)
            return self._encode_binary(rst)
        else:
            return self._encode_binary(False)

    def _feature_minmax_item_age_shop(self, target_age, user_id, shop_id):
        """ Checks if the total of the item
        is greater than the minimal age for this user by shop and
        less than the maximal age for this user by shop, i.e.
        min_age < age < max_age.
        """
        if (user_id, shop_id) in self.stats_by_user_shop.index:
            min_age = self.stats_by_user_shop.ix[(user_id, shop_id), 'item_age_min']
            max_age = self.stats_by_user_shop.ix[(user_id, shop_id), 'item_age_max']
            rst = (min_age < target_age) & (target_age < max_age)
            return self._encode_binary(rst)
        else:
            return self._encode_binary(False)

    #################

    def _feature_is_date_not_in_future(self):

        df_row = self.target_row
        if df_row['date'] == MISSING_DATE:
            return False
        else:
            return self._encode_binary(df_row['created_at'] >= df_row['date'])
    #################

    def _feature_vector(self, target_row):
        # The next self.members to be used in the feature functions
        """ Checks if the total of the item
        is greater than the minimal total for this user and shop and
        less than the maximal total for this user , i.e.
        min_total < total < max_total.
        """

        user = target_row['user_id']
        shop = target_row['shop_id']
        currency = target_row['currency_id']

        item_age = (target_row['date'] - target_row['created_at'])/SECONDS_PER_DAY

        target_total = target_row['total_amount']

        self.target_row = target_row

        feature_arr = np.asarray([
            (self._feature_user_has_supplier(user, shop)),
            (self._feature_user_has_currency(user, currency)),
            (self._feature_avg_total_range(target_total, user)),
            (self._feature_avg_total_range_shop(target_total, user, shop)),
            (self._feature_total_minmax(target_total, user)),
            (self._feature_total_minmax_shop(target_total, user, shop)),
            (self._feature_avg_item_age(item_age, user)),
            (self._feature_avg_item_age_shop(item_age, user, shop)),
            (self._feature_minmax_item_age(item_age, user)),
            (self._feature_minmax_item_age_shop(item_age, user, shop)),
            (self._feature_is_date_not_in_future())
        ])

        return feature_arr
