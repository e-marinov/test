from confidence import field_convert_map, read_dataset, get_column_names
from generate_targets import field_compare_map

import pandas as pd

import argparse
import logging
import sys

import os



def get_fields_to_compare(args, fields_01, fields_02):

    logging.debug('Inside get_fields_to_compare function...')

    fields_rst = []

    if len(args.fields) == 1 and args.fields[0] == 'all':
        if set(fields_01) != set(fields_02):
            exit_message = ('The column names in %s and %s are NOT equal, \
                            provided thath fields = all.'
                            % (args.targets[0], args.targets[1]))
            raise SystemExit(exit_message)
            return []

        else:
            exit_message = ('The column names in %s and %s are equal.'
                            % (args.targets[0], args.targets[1]))
            fields_rst = fields_01
    else:
        if not (set(args.fields) <= set(fields_01)):
            exit_message = ('The fields from the command line are NOT a \
                            subset of the column names in %s.'
                            % args.targets[0])
            raise SystemExit(exit_message)
            return []
        elif not (set(args.fields) <= set(fields_02)):
            exit_message = ('The fields from the command line are NOT a \
                            subset of the column names in %s.'
                            % args.targets[1])
            raise SystemExit(exit_message)
            return []
        else:
            logging.info('The fields from the command line are a subset\
                         of the column names in %s and in %s as well.'
                         % (args.targets[1], args.targets[1]))
            fields_rst = args.fields

    return fields_rst


def compare_records(row01, row02, dict_diff):

    for fld in dict_diff.keys():
        dict_diff[fld] = field_compare_map[fld](row01[fld], row02[fld])


def compare_data_sets(ds01, ds02):
    logging.debug('Inside compare_data_sets...')
#    result = True
    diff_list = []
    if len(set(ds01.columns) ^ set(ds02.columns)) > 0:
        logging.critical('The column names of the data sets are NOT equal...')
#        return False

    if len(ds02) != len(ds02):
        logging.critical('The number of rows of the first and second target \
                          file are NOT equal...')
#        return False
    dict_diff = dict.fromkeys(ds01.columns)

    for idx in range(len(ds01)):
        row01 = ds01.iloc[idx]
        row02 = ds02.iloc[idx]

        for fld in ds01.columns:
            dict_diff[fld] = field_compare_map[fld](row01[fld], row02[fld])

        diff_values = dict_diff.values()
        if False in diff_values:
            index_loc = [ds01.index[idx], ds02.index[idx]]
            diff_list.append(tuple(index_loc + diff_values))

    return (['id01', 'id02'] + dict_diff.keys(), diff_list)


def get_target_conversions(fields):

    target_conversions = {}

    for fld in fields:
        try:
            target_conversions[fld] = field_convert_map[fld]
        except KeyError:
            target_conversions[fld] = str

    return target_conversions

def main():

    parser = argparse.ArgumentParser(description=
                                     'Compare two csv files by fields.')

    parser.add_argument('targets', metavar='target_files', type=str,
                        nargs=2,
                        help='csv files to be compared: \
                        target01.csv target02.csv')
    # fields_default = ['shop_id', 'currency_code', 'total_amount', 'user_id']
    fields_default = ['all']
    parser.add_argument('--fields', default=fields_default, nargs='*',
                        type=str,
                        help='fields to be taken into account: shop_id, \
                        total amount, and all (as default) to compare \
                        all fields')
    parser.add_argument('--log', default=['WARNING'], nargs=1, type=str,
                        help='set loglevel: DEBUG, INFO, WARNING, CRITICAL')

    args = parser.parse_args()
#    args = parser.parse_args('join_target_new02.csv targets_conv01orig.csv \
#    --fields target_val shop_id currency_code total_amount --log info'.split())

    print 'Before logging...'
    logging.basicConfig(level=getattr(logging, args.log[0].upper()))
    logging.info('Start compare target script...')

    dir_working = os.getcwd()
    target01_path = dir_working + '/' + args.targets[0]
    target02_path = dir_working + '/' + args.targets[1]

    logging.info('First target file: %s' % target01_path)
    logging.info('Second target file: %s' % target02_path)

    fields01 = get_column_names(target01_path)
    fields02 = get_column_names(target02_path)
    print fields01
    print fields02
    fields = get_fields_to_compare(args, fields01, fields02)
    logging.info('Fields to be examined: %s' % fields)

    conv_map = field_convert_map    
    conversions = dict((k, conv_map[k]) for k in conv_map if k in fields)

    df_target01 = read_dataset(target01_path, conversions)[fields]
    df_target02 = read_dataset(target02_path, conversions)[fields]

    # list to hold the indeces from target01 and target02 which differs
    target_fields, diff_list = compare_data_sets(df_target01, df_target02)

    if len(diff_list) > 0:
        print('There are differences in the target files.')
        report = pd.DataFrame(diff_list, columns=target_fields)
        report.to_csv('report_target.csv', index=False)
    else:
        print('The target files are equal.')

if __name__ == "__main__":
    main()
