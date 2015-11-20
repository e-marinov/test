import data_io
from tokenization import Parser
from transforms import ExtractionModel
from total_amount_features import TotalAmountFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
import sys
import decimal
import pickle


feature_mappers = {
    'total_amount': TotalAmountFeatures
}

classifiers = {
    'total_amount': RandomForestClassifier(n_estimators=100, random_state=1)
}


def train_model(data, var_name):
    targets = [decimal.Decimal("%.2f" % v) for v in data[var_name]]
    model = ExtractionModel(Parser,
                            feature_mappers[var_name],
                            classifiers[var_name])
    model.fit(data, targets)
    return model


def print_usage():
    print('Usage: {} <dataset path> <models file> <list of fields>')


def main():
    if len(sys.argv) < 3:
        print_usage()
        return

    data = data_io.load_from_files(sys.argv[1])
    fields = sys.argv[3:]
    models_fname = sys.argv[2]

    try:
        with open('models.p') as model_file:
            models = pickle.load(model_file)
    except:
        models = {}

    print 'before training.........'    
    
    for field in fields:
        models[field] = train_model(data, field)
        print 'field: ', field

    with open(models_fname, 'w') as model_file:
        pickle.dump(models, model_file)


if __name__ == "__main__":
    main()
