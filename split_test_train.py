import random
import shutil
import pandas as pd
import os

import string


data = pd.read_csv('/home/evgeniy/ReceiptBank/rb-engine/csvs/user_receipts.csv')

NMB_TRAIN = 120000
NMB_TEST = 30000

MAX_LINES_NMB = 100

def copy(src, dst, files):

    for f in files:
        src_file = os.path.join(src, f)
        dst_file = os.path.join(dst, f)
        shutil.copyfile(src_file, dst_file)


def to_number(file_name):
    return string.atoi(string.split(file_name, '.')[0])

path = '/home/evgeniy/ReceiptBank/rb-engine/rb_engine_dataset'

path_split = '/home/evgeniy/ReceiptBank/rb-engine/rb_digital_split'
path_test = os.path.join(path_split, 'test_large30')
path_train = os.path.join(path_split, 'train_large120')


files = os.listdir(path)
random.shuffle(files)
files_sample = []
for f in files:
    with open(os.path.join(path, f)) as foo:
        n_lines = len(foo.readlines())
    if n_lines < MAX_LINES_NMB:
        files_sample.append(f)
    else:
        continue
    if len(files_sample) == NMB_TEST + NMB_TRAIN:
        break

# files_sample = random.sample(files, 50000)
train_files = random.sample(files_sample, NMB_TRAIN)
test_files = set(files_sample) - set(train_files)

ids_test = map(to_number, test_files)
ids_train = map(to_number, train_files)


data[data['id'].isin(ids_test)].to_csv(
    os.path.join(path_test, 'data.csv'), index=False)
data[data['id'].isin(ids_train)].to_csv(
    os.path.join(path_train, 'data.csv'), index=False)


copy(path, path_test, test_files)
copy(path, path_train, train_files)
