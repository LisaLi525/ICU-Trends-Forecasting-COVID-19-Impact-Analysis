import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    df.drop(['bc_m/s/t_active'], axis=1, inplace=True)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index(['date'], drop=True)
    return df

def plot_data(df):
    plt.figure(figsize=(10, 6))
    df['bc_icu_active'].plot()
    plt.show()

def prepare_data(df, split_date):
    df = df['bc_icu_active']
    train = df.loc[:split_date]
    test = df.loc[split_date:]

    plt.figure(figsize=(10, 6))
    ax = train.plot()
    test.plot(ax=ax)
    plt.legend(['train', 'test'])
    plt.show()

    return train, test

def scale_data(train, test):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_sc = scaler.fit_transform(train.values.reshape(-1, 1))
    test_sc = scaler.transform(test.values.reshape(-1, 1))

    return train_sc, test_sc

def create_sequences(data, label_name):
    df = pd.DataFrame(data, columns=[label_name])
    for s in range(1, 2):
        df['X_{}'.format(s)] = df[label_name].shift(s)
    df = df.dropna()
    X = df.drop(label_name, axis=1).values
    y = df[label_name].values
    return X, y

def build_nn_model(X_train, y_train):
    nn_model = Sequential()
    nn_model.add(Dense(12, input_dim=1, activation='relu'))
    nn_model.add(Dense(1))
    nn_model.compile(loss='mean_squared_error', optimizer='adam')
    early_stop = EarlyStopping(monitor='loss', patience=2, verbose=1)
    nn_model.fit(X_train, y_train, epochs=100, batch_size=1, verbose=1, callbacks=[early_stop], shuffle=False)
    return nn_model

def build_lstm_model(X_train, y_train):
    X_train_lmse = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    lstm_model = Sequential()
    lstm_model.add(LSTM(7, input_shape=(1, X_train_lmse.shape[1]), activation='relu', kernel_initializer='lecun_uniform', return_sequences=False))
    lstm_model.add(Dense(1))
    lstm_model.compile(loss='mean_squared_error', optimizer='adam')
    early_stop = EarlyStopping(monitor='loss', patience=2, verbose=1)
    lstm_model.fit(X_train_lmse, y_train, epochs=100, batch_size=1, verbose=1, shuffle=False, callbacks=[early_stop])
    return lstm_model, X_train_lmse

def evaluate_model(model, X_test, y_test, model_type="NN"):
    test_mse = model.evaluate(X_test, y_test, batch_size=1)
    print(f'{model_type}: {test_mse}')

def plot_predictions(model, X_test, y_test, title):
    y_pred_test = model.predict(X_test)
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label='True')
    plt.plot(y_pred_test, label=title)
    plt.title(f"{title}'s Prediction")
    plt.xlabel('Observation')
    plt.ylabel('ICU')
    plt.legend()
    plt.show()

def main():
    filepath = 'path/to/your/data.csv'
    df = load_and_preprocess_data(filepath)
    plot_data(df)
    split_date = pd.Timestamp('2021-05-01')
    train, test = prepare_data(df, split_date)
    train_sc, test_sc = scale_data(train, test)

    X_train, y_train = create_sequences(train_sc, 'Y')
    X_test, y_test = create_sequences(test_sc, 'Y')

    nn_model = build_nn_model(X_train, y_train)
    evaluate_model(nn_model, X_test, y_test)

    lstm_model, X_train_lmse = build_lstm_model(X_train, y_train)
    X_test_lmse = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    evaluate_model(lstm_model, X_test_lmse, y_test, model_type="LSTM")

    plot_predictions(nn_model, X_test, y_test, "ANN")
    plot_predictions(lstm_model, X_test_lmse, y_test, "LSTM")

if __name__ == "__main__":
    main()
