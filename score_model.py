import data_io
import pandas as pd
import numpy as np
import decimal
import sys
import pickle


def score_dataset(data, model_fname, var_name):
    with open(model_fname) as model_file:
        models = pickle.load(model_file)

    res = models[var_name].score(data, data[var_name])
    print "Score: %.4f" % res


def print_usage():
        print('Usage: {} <dataset path> <model filename> <list of fields>')


def main():
    if len(sys.argv) < 4:
        print_usage()
        return

    data = data_io.load_from_files(sys.argv[1])
    model_fname = sys.argv[2]
    fields = sys.argv[3:]

    for field in fields:
        score_dataset(data, model_fname, field)

if __name__ == "__main__":
    main()
