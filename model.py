from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from pickle import dump, load
import pandas as pd
import numpy as np

def load_data():
    data = dict()
    tables = ('clients',
            'close_loan',
            'job',
            'last_credit',
            'loan',
            'pens',
            'salary',
            'target',
            'work')
    for table in tables:
        data[table] = pd.read_csv('data/D_' + table + '.csv')
    return data

def join_data(data):
    df = data['target']
    df = df.merge(data['clients'],
             left_on='ID_CLIENT',
             right_on='ID', how='left')
    df = df.drop(columns=['ID'])
    df = df.merge(data['salary'],
             left_on='ID_CLIENT',
             right_on='ID_CLIENT', how='left')
    df = df.merge(
        data['loan'].groupby(['ID_CLIENT']).agg(LOAN_NUM_TOTAL=('ID_LOAN',
                                                                'count')),
        left_on='ID_CLIENT',
        right_on='ID_CLIENT', how='left')
    closed_per_client = data['loan'].merge(
        data['close_loan'],
        left_on='ID_LOAN',
        right_on='ID_LOAN', how='left').groupby(['ID_CLIENT']).agg(LOAN_NUM_CLOSED=('CLOSED_FL',
                                                                'count'))
    df = df.merge(
        closed_per_client,
        left_on='ID_CLIENT',
        right_on='ID_CLIENT', how='left')
    df = df.drop(columns=['ID_CLIENT',
                          'REG_ADDRESS_PROVINCE',
                          'POSTAL_ADDRESS_PROVINCE',
                          'FACT_ADDRESS_PROVINCE'])
    # df = pd.get_dummies(df, columns=['EDUCATION', 'MARITAL_STATUS', 'FAMILY_INCOME'])
    return df

def open_data():
    return join_data(load_data())

def preprocess_data(df: pd.DataFrame, target=False):

    if target:
        X_df, y_df = split_data(df)
    else:
        X_df = df
    
    features_whitelist = ['AGE', 'GENDER', 'MARITAL_STATUS', 'EDUCATION']
    X_df = X_df.loc[:, features_whitelist]
    #print('before encode', X_df.columns)

    to_encode = ['EDUCATION', 'MARITAL_STATUS', 'GENDER']
    for col in to_encode:
        dummy = pd.get_dummies(X_df[col], prefix=col)
        X_df = pd.concat([X_df, dummy], axis=1)
        #print('after concat', X_df.columns)
        X_df.drop(col, axis=1, inplace=True)
        #print('after drop', X_df.columns)

    X_df.dropna(inplace=True)
    #print(X_df.head(1))

    if target:
        return X_df, y_df
    else:
        return X_df

def split_data(df: pd.DataFrame):
    y = df['TARGET']
    X = df.drop(columns=['TARGET'])

    return X, y


def fit_and_save_model(X_df, y_df, path="data/model_weights.mw"):
    model = RandomForestClassifier()
    model.fit(X_df, y_df)

    test_prediction = model.predict(X_df)
    accuracy = accuracy_score(test_prediction, y_df)
    print(f"Model accuracy is {accuracy}")

    with open(path, "wb") as file:
        dump(model, file)

    print(f"Model was saved to {path}")


def load_model(path="data/model_weights.mw"):
    with open(path, "rb") as file:
        return load(file)


def predict(model, df):
    prediction = model.predict(df)[0]
    #print('not squized', prediction)
    #prediction = np.squeeze(prediction)
    #print('squized', prediction)

    prediction_proba = model.predict_proba(df)
    #print('======= not squized proba', prediction_proba)
    prediction_proba = np.squeeze(prediction_proba)
    #print('======= squized proba', prediction_proba)

    # encode_prediction_proba = {
    #     0: "Вам не понравится наше предложение с вероятностью",
    #     1: "Вам понравится наше предложение с вероятностью"
    # }

    encode_prediction = {
        0: "Боимся, Вам не будет интересно наше предложение 😔",
        1: "Кажется, Вам понравится наше предложение! 😊"
    }

    # prediction_data = {}
    # for key, value in encode_prediction_proba.items():
    #     prediction_data.update({value: prediction_proba[key]})

    return encode_prediction[prediction], f'Уверены в этом с вероятностью в {int(prediction_proba[prediction] * 100)}%'


if __name__ == "__main__":
    df = open_data()
    X_df, y_df = preprocess_data(df, target=True)
    fit_and_save_model(X_df, y_df)