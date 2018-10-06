import pandas as pd

def clean_data(data_path):
    data = pd.read_csv(data_path, index_col='PassengerId')

    data = data.rename(columns={'Pclass': 'Class',
                                'SibSp': 'Siblings',
                                'Parch': 'Parents'})

    data['Sex'].replace(['female', 'male'], [0, 1], inplace=True)

    CLASS_ONEHOT = pd.get_dummies(data['Class'])

    CLASS_ONEHOT = CLASS_ONEHOT.rename(columns={1: 'First',
                                                2: 'Second',
                                                3: 'Third'})

    data = data.join(CLASS_ONEHOT)

    x = data.drop(columns=['Name',
                           'Fare',
                           'Class',
                           'Cabin',
                           'Age',
                           'Embarked',
                           'Ticket'])

    return x
