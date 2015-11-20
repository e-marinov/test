import data_io
from transforms import ParserTransform
from tokenization import Parser
import decimal
import datetime
import sys


def score_dataset(data):
    t = ParserTransform(Parser)
    t.fit(data)
    date_hits = 0
    total_amount_hits = 0
    for parser, (item_id, item) in zip(t.transform(data), data.iterrows()):
        values = [token.value for token in parser.date_tokens]
        if item['date'] in values:
            date_hits += 1

        values = [token.value for token in parser.numeric_tokens]
        if decimal.Decimal("%.2f" % item['total_amount']) in values:
            total_amount_hits += 1

    print("Total amount recall %d (%.4f)" %
          (total_amount_hits, (float(total_amount_hits)/len(data))))
    print("Date recall %d (%.4f)" % (date_hits, (float(date_hits)/len(data))))


def main():
    if len(sys.argv) > 1:
        data = data_io.load_from_files(sys.argv[1])
        score_dataset(data)
    else:
        print("Usage: {} <dataset path>".format(sys.argv[0]))


if __name__ == "__main__":
    main()
