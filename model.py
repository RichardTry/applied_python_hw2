from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from pickle import dump, load
import pandas as pd

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

def preprocess_data(df: pd.DataFrame, test=True):
    df.dropna(inplace=True)

    if test:
        X_df, y_df = split_data(df)
    else:
        X_df = df

    to_encode = ['EDUCATION', 'MARITAL_STATUS', 'FAMILY_INCOME']
    for col in to_encode:
        dummy = pd.get_dummies(X_df[col], prefix=col)
        X_df = pd.concat([X_df, dummy], axis=1)
        X_df.drop(col, axis=1, inplace=True)

    if test:
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


def load_model_and_predict(df, path="data/model_weights.mw"):
    with open(path, "rb") as file:
        model = load(file)

    prediction = model.predict(df)[0]
    # prediction = np.squeeze(prediction)

    prediction_proba = model.predict_proba(df)[0]
    # prediction_proba = np.squeeze(prediction_proba)

    encode_prediction_proba = {
        0: "Вам не понравится наше предложение с вероятностью",
        1: "Вам понравится наше предложение с вероятностью"
    }

    encode_prediction = {
        0: "Боимся, нам не будет интересно наше предложение :(",
        1: "Кажется, Вам понравится наше предложение!"
    }

    prediction_data = {}
    for key, value in encode_prediction_proba.items():
        prediction_data.update({value: prediction_proba[key]})

    prediction_df = pd.DataFrame(prediction_data, index=[0])
    prediction = encode_prediction[prediction]

    return prediction, prediction_df


if __name__ == "__main__":
    df = open_data()
    X_df, y_df = preprocess_data(df)
    fit_and_save_model(X_df, y_df)