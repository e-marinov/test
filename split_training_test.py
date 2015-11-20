import numpy as np
import pandas as pd

import argparse
import os
import sys


def parse_arguments():
    parser = argparse.ArgumentParser(description='Split data into training and\
    test data.')
    parser.add_argument('--percent', metavar='targets.csv',
                        type=int, nargs=1, default=[25],
                        help='Persentage of the test data.')
    parser.add_argument('--historic-receipts', metavar='historic_receipts.csv',
                        type=str, nargs=1, default=['historic_receipts.csv'],
                        help='receipts with target values used for training')

    parser.add_argument('--current-receipts', metavar='receipts.csv',
                        type=str, nargs=1, default=['receipts.csv'],
                        help='receipts with correct values used for statistics')

    parser.add_argument('--count', metavar='<number>',
                        type=int, nargs=1, default=[5000],
                        help='number of receipts in training')

    parser.add_argument('--train', metavar='model.p',
                        type=str, nargs=1, default=[''],
                        help='create a trained model file')

    return parser.parse_args()



def main():
    if len(sys.argv) < 3:
        print_usage()
        return

    data = data_io.load_from_files(sys.argv[1])
    fields = sys.argv[2:]

    try:
        with open('models.p') as model_file:
            models = pickle.load(model_file)
    except:
        models = {}

    print 'before training.........'    
    
    for field in fields:
        models[field] = train_model(data, field)
        print 'field: ', field

    with open('models.p', 'w') as model_file:
        pickle.dump(models, model_file)


if __name__ == "__main__":
    main()
