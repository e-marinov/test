from confidence import read_dataset

import pandas as pd
import numpy as np

import argparse
import logging

import os
import sys

def str_to_int(x):
    if x == '':
        return ''
    else:
        return str(int(float(x.strip())))

def str_to_float(x):
    if x == '':
        return ''
    else:
        return str(float(x.strip()))

convert_map = {
    'INT': str_to_int,
    'FLOAT': str_to_float
    }


def indices(lst, element, start = 0):
    result = []
    offset = start - 1
    while True:
        try:
            offset = lst.index(element, offset+1)
        except ValueError:
            return result
        result.append(offset)

def get_convertion_fields(lst, conv_types, start=0):

    result = []    
    for fld in lst:
        if fld in conv_types:
            return result
        else:
            result.append(fld)
            

def main():
    print('main starting...')
    descr = 'Convert column type from csv files.'
    parser = argparse.ArgumentParser(description=descr)

    parser.add_argument('fname', metavar='fname.csv', type=str,
                        nargs=1,
                        help='set the name of the input file: target.csv')
    parser.add_argument('--converted_fname', metavar='converted_target.csv', 
                        type=str,
                        nargs=1,
                        help='set the name of the converted file:\
                        converted.csv')
    parser.add_argument('--convert', nargs='*', type=str,
                        help='convert column types: int shop_id\
                        float field01 field02 int field03')
    parser.add_argument('--log', default=['INFO'], nargs=1, type=str,
                        help='set loglevel: DEBUG, INFO, WARNING, CRITICAL')

    args = parser.parse_args('receipt_events.csv \
    --converted_fname conv_receipt_events.csv --log info \
    --convert  int user_id shop_id float total_amount'.split())
    
#    args = parser.parse_args(sys.argv)
    if len(args.converted_fname) == 0:
        args.converted_fname.append(args.fname[0])
        
    logging.basicConfig(level=getattr(logging, args.log[0].upper()))

#    if len(args.convert) % 2 != 0:
#        logging.warning('Convert fields are not correctly set.')
#        return
    print args.convert
    
    types = convert_map.keys()
    
    idx_holder = dict((k, []) for k in types)
        
    typ = args.convert[0].upper()
    if typ not in types:
        logging.CRITICAL('Argument from the Command line are not correct.')
        logging.CRITICAL('%s is not a correct convert type.' % typ)
    
    for item in args.convert[1:]:
        
        if item.upper() in types:
            typ = item.upper()
        else:
            idx_holder[typ].append(item)

    print idx_holder
    # Check if there are not same fields for different conversion types
    for i01, item01 in enumerate(types[0:-1]):
        for item02 in types[i01+1:]:
            if len(set(idx_holder[item01]) & set(idx_holder[item02])) > 0:
                logging.CRITICAL('Every field should be converted into one type')
                return
    # Remove the fields which occur twice
    for typ in types:
        idx_holder[typ] = list(set(idx_holder[typ]))
   

    dir_working = os.getcwd()
    file_path = dir_working + '/' + args.fname[0]
    
    fields = []
    logging.debug('Read the column names from the target file...')
    with open(file_path) as f_01:
        line = f_01.next()
        fields = line.strip().split(',')
    f_01.close()
    
    conversions_str = dict((k, str) for k in fields)    
    
    dataset = pd.read_csv(file_path, converters=conversions_str)

    for typ, fld_list in idx_holder.iteritems():
        for fld in fld_list:
            dataset[fld] = map(convert_map[typ], dataset[fld])
        
#    for i in range(0, len(args.convert), 2):
#        fields_dict[args.convert[i]] = convert_map[args.convert[i+1].upper()]
#        
    dataset.to_csv(args.converted_fname[0], index=False)    
    
    print('End of main...')


if __name__ == "__main__":
    main()
