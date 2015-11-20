""" Common functions to deal with datasets.

Includes procedures to extract a dataset from the production database,
and to store and load a dataset as a set of files.
"""
import sys
import os
import glob
import dateutil
import decimal
import pickle
import yaml
import sqlalchemy
from sqlalchemy import select
from sqlalchemy.sql import compiler
import psycopg2
from psycopg2.extensions import adapt as sqlescape
import pandas as pd
import numpy as np


def root_path():
    if os.path.isabs(__file__):
        path = os.path.dirname(__file__)
    else:
        path = os.path.dirname(os.getcwd() + os.path.sep + __file__)
    return path


def database_url():
    """ Return a URL-style string that can be passed to SQLAlchemy """
    conf_file = os.path.sep.join([root_path(), 'database.yml'])
    params = {'adapter': 'postgresql',
              'username': 'rbank',
              'database': 'rbank',
              'password': '',
              'host': 'localhost',
              'port': 5432}

    try:
        with open(conf_file) as f:
            params.update(yaml.load(f))
    except:
        pass

    config = "{adapter}://{username}:{password}@{host}:{port}/{database}"
    return config.format(**params)


def init_database():
    """ Initializes and returns SQLAlchemy database connection """
    metadata = sqlalchemy.MetaData()
    metadata.bind = sqlalchemy.create_engine(database_url(), encoding='UTF-8')
    metadata.reflect()
    return metadata


def compile_query(metadata, query):
    """ Convert an SQLAlchemy query to an SQL command

    SQLAlchemy offers a nice syntax to build SQL queries, but since we
    want to extract the data with pandas so as to get a dataframe object,
    this helper method is needed to convert the query object to SQL
    command string that can be used by pandas.
    """
    comp = compiler.SQLCompiler(metadata.bind.dialect, query)
    enc = metadata.bind.dialect.encoding
    params = {}
    for k, v in comp.params.iteritems():
        if isinstance(v, unicode):
            v = v.encode(enc)
        params[k] = sqlescape(v)
    return (comp.string.encode(enc) % params).decode(enc)


def load_from_database(metadata, date_from=None, date_to=None, limit=None,
                       image_types=None, receipt_id=None, include_text=True):
    """ Create a dataset by querying the production database

    If a receipt_id is given then only that receipt is extracted, otherwise
    a full dataset is extracted.
    """
    receipts = metadata.tables['receipts']
    shops = metadata.tables['shops']
    ocr_datas = metadata.tables['ocr_datas']

    fields = [receipts.c.id, receipts.c.total_amount, receipts.c.vat_amount,
              receipts.c.date, receipts.c.currency_code,
              receipts.c.invoice_number, receipts.c.image_content_type,
              receipts.c.shop_id, receipts.c.user_id,
              shops.c.name.label('shop_name')]
    if include_text:
        fields.append(ocr_datas.c.text)

    if type(receipt_id) is int:
        cond = receipts.c.id == receipt_id
    elif receipt_id is not None:
        cond = receipts.c.id.in_(np.asarray(receipt_id))
    else:
        cond = ~receipts.c.is_merged
        cond &= receipts.c.state == 'complete'
        cond &= receipts.c.total_amount != 0
        cond &= receipts.c.date != None
        if date_from is not None:
            cond &= receipts.c.created_at > date_from
        if date_to is not None:
            cond &= receipts.c.created_at < date_to

        if image_types is not None:
            cond &= receipts.c.image_content_type.in_(image_types)

    query = select(fields, cond).\
            select_from(receipts.
                        join(shops, shops.c.id == receipts.c.shop_id).
                        join(ocr_datas, ocr_datas.c.receipt_id == receipts.c.id))

    if limit is not None:
        query = query.limit(limit)

    q = compile_query(metadata, query)
    df = pd.io.sql.read_sql(q, metadata.bind.raw_connection(), index_col='id')
    df['shop_name'] = [name.encode('UTF-8') for name in df['shop_name']]
    return df


def save_to_files(data, dir_name):
    """ Save a dataset as a set of files

    This saves the OCR text of each receipt as a separate text file and
    a CSV file containing the receipt fields. This can be later loaded
    back to a DataFrame by calling 'load_from_files'.

    Parameters
    ----------
    data: DataFrame
        The dataset to be saved.

    dir_name: string
        Path to the directory where the files should be created.
    """
    try:
        os.mkdir(dir_name)
    except:
        for p in glob.glob('%s/*.txt' % dir_name):
            os.remove(p)

    for idx, item in data.iterrows():
        with open('%s/%d.txt' % (dir_name, idx), 'w') as f:
            if item['text'] is not None:
                f.write(item['text'].encode('UTF-8'))

    tmp = data[data.columns - ['text']]
    tmp.to_csv('%s/data.csv' % dir_name, encoding='UTF-8')


def read_file(path):
    with open(path) as f:
        txt = f.read()
    return txt


def load_from_files(dir_name):
    """ Load a dataset stored as a set of files

    Loads a dataset created by 'save_to_files'

    Parameters
    ----------
    dir_name: string
        Path to root directory of the files.
    """

    data = pd.read_csv(os.path.join(dir_name, 'data.csv'), index_col='id')
#    data['date'] = [dateutil.parser.parse(x).date() for x in data['date']]

    data['total_amount'] = [decimal.Decimal('%.2f' % x)
                            for x in data['total_amount']]
    data['vat_amount'] = [decimal.Decimal('%.2f' % x)
                          for x in data['vat_amount']]

    paths = glob.glob('%s/*.txt' % dir_name)
    ids = [int(os.path.splitext(os.path.basename(p))[0]) for p in paths]
    data = data.ix[ids]

    texts = [read_file('%s/%d.txt' % (dir_name, idx)) for idx in data.index]
    data['text'] = texts
    return data
