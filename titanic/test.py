import sys
from keras.models import load_model
from utils import clean_data

SUBMISSION = sys.argv[1]

model = load_model(f"models/{SUBMISSION}.h5")

X_test = clean_data('data/test.csv')

result = model.predict(X_test)

X_test['Survived'] = result.round().astype(int)

for column in X_test.columns:
    if column != 'Survived':
        X_test = X_test.drop(columns=column)

X_test.to_csv(f"output/{SUBMISSION}.csv")
