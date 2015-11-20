from confidence_features import MISSING_ID, MISSING_DATE

import numpy as np
import pandas as pd
import operator as op

import argparse
import logging

import os
import sys


def equals_float(x, y):
    return abs(x - y) < 0.001


def equals_id(x, y):
    return (x == y) & (x != MISSING_ID) & (y != MISSING_ID)


def equals_date(x, y):
    return (x == y) & (x != MISSING_DATE) & (y != MISSING_DATE)


def less_than_date(x, y):
    return (x < y) & (x != MISSING_DATE) & (y != MISSING_DATE)


# This is a list of the output columns that will be computed, the assumption
# is that each output column is computed by a binary operation taking one
# column from the receipts table and one from the events table. So each
# output is described by four things:
#  - name of the output column
#  - name of the receipts column
#  - name of the events column
#  - the binary operation to be used
fields_to_compute = [
    ['shop_id_eq', 'shop_id', 'shop_id', equals_id],
    ['currency_id_eq', 'currency_id', 'currency_id', equals_id],
    ['date_eq', 'date', 'date', equals_date],
    ['total_amount_eq', 'total_amount', 'total_amount', equals_float],
    ['future_date', 'created_at', 'date', less_than_date]
]


# The following two lists describe which columns of the receipt and events
# sets should be copied to the generated training set file.
fields_from_receipts = ['user_id', 'created_at']
fields_from_events = ['type_id', 'shop_id', 'total_amount', 'vat_amount',
                      'date', 'currency_id']


def read_receipts_dataset(path):
    receipts = pd.read_csv(path, index_col='id')
    return receipts


def read_events_dataset(path, event_type):
    events = pd.read_csv(path, index_col='id')
    events = events[events['event_type'] == event_type]
    events = events.set_index(['receipt_idx'])
    return events


def create_targets(receipts, events, target_fields):
    """ Merge current and historical data to produce a training set

    The information in 'receipts' should contain current (i.e. "correct") data
    for the receipts. The 'events' dataframe on the other hand contains
    historical data when a specific kind of event has occurred. By comparing
    the state at those two points we can determine whether the receipt has been
    correct at the time of the event, which is used as the target for training
    a classifier.
    """

    valid_rows = receipts.index & events.index
    result = pd.DataFrame(index=valid_rows)
    result.index.name = 'id'

    for output_name, receipts_column, events_column, op in fields_to_compute:
        result[output_name] = op(receipts.ix[valid_rows, receipts_column],
                                 events.ix[valid_rows, events_column])

    result['target_val'] = np.all(result[target_fields], axis=1)
    logging.info('The fields of result are: %s' % result.columns)
    return result


def create_historic_data(receipts, events):
    valid_rows = receipts.index & events.index
    result = pd.DataFrame(index=valid_rows)
    result.index.name = 'id'

    for field in fields_from_receipts:
        result[field] = receipts.ix[valid_rows, field]

    for field in fields_from_events:
        result[field] = events.ix[valid_rows, field]

    return result


def parse_arguments():
    parser = argparse.ArgumentParser(description='Produce a target file.')

    parser.add_argument('--targets', metavar='targets_fname.csv', type=str,
                        nargs=1,
                        default=['targets.csv'],
                        help='set the name of the targets file')
    parser.add_argument('--historic', metavar='historic_fname.csv', type=str,
                        nargs=1,
                        default=['historic_receipts.csv'],
                        help='set the name of the historic file')
    parser.add_argument('--receipts', default=['receipts.csv'], nargs=1,
                        type=str,
                        help='set receipts file: receipts.csv')
    parser.add_argument('--events', metavar='receipt_events.csv event_type',
                        default=['receipt_events.csv', 'rb_engine'],
                        nargs=2, type=str,
                        help='set events file and type: receipt_events.csv \
                        rb_enine...')
    parser.add_argument('--log', default=['INFO'], nargs=1, type=str,
                        help='set loglevel: DEBUG, INFO, WARNING, CRITICAL')

    fields_sum_default = ['shop_id_eq', 'currency_id_eq', 'total_amount_eq']
    parser.add_argument('--target_fields', default=fields_sum_default, nargs='*',
                        type=str,
                        help='set fields to be summarized in the target_val\
                        column.')
    return parser.parse_args()


def main():
    args = parse_arguments()
    logging.basicConfig(level=getattr(logging, args.log[0].upper()))

    logging.info('Generating target file script starting...')

    logging.info('INFO....')

    dir_working = os.getcwd()
    print args.receipts, type(args.receipts[0])

    receipts_path = dir_working + '/' + args.receipts[0]
    receipt_ev_path = dir_working + '/' + args.events[0]
    event_type = args.events[1]

    logging.debug('Reading receipts: %s' % receipts_path)
    receipts = read_receipts_dataset(receipts_path)
    print args.events[0]
    logging.info('Reading events: %s, %s' % (receipt_ev_path, event_type))
    events = read_events_dataset(receipt_ev_path, event_type)
    # Treat the fields
    dataset = create_targets(receipts, events, args.target_fields)

    logging.debug('Target dataset keys: %s' % dataset.keys())

    # Compute the number of the True and False values
    samples_count = len(dataset)
    positive_count = np.count_nonzero(dataset['target_val'])
    negative_count = samples_count - positive_count

    print 'Length of target array', samples_count
    print 'Count True values', positive_count, float(positive_count)/samples_count
    print 'Count False values', negative_count, float(negative_count)/samples_count

    dataset.to_csv(args.targets[0])

    dataset = create_historic_data(receipts, events)
    dataset.to_csv(args.historic[0])

    logging.info('Generating target file script finishing...')


if __name__ == "__main__":
    main()
