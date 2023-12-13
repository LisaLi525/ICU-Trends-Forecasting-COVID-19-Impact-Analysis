library(pandas)
library(numpy)
library(matplotlib)
library(keras)
library(sklearn)
library(tensorflow)

# Function to load data
load_data <- function(filepath) {
  pd.read_csv(filepath)
}

# Function to preprocess data
preprocess_data <- function(df) {
  df['date'] = pd.to_datetime(df['date'])
  df = df.set_index(['date'], drop=True)
  return(df)
}

# Function to split data
split_data <- function(df, split_date) {
  train = df.loc[:split_date]
  test = df.loc[split_date:]
  return(list(train, test))
}

# Function to scale data
scale_data <- function(train, test) {
  scaler = MinMaxScaler(feature_range=(-1, 1))
  train_sc = scaler.fit_transform(train.values.reshape(-1, 1))
  test_sc = scaler.transform(test.values.reshape(-1,1))
  return(list(train_sc, test_sc))
}

# Function to create ANN model
create_ann_model <- function(input_shape) {
  nn_model = Sequential()
  nn_model.add(Dense(12, input_dim=input_shape, activation='relu'))
  nn_model.add(Dense(1))
  nn_model.compile(loss='mean_squared_error', optimizer='adam')
  return(nn_model)
}

# Function to create LSTM model
create_lstm_model <- function(input_shape) {
  lstm_model = Sequential()
  lstm_model.add(LSTM(7, input_shape=(1, input_shape), activation='relu', kernel_initializer='lecun_uniform', return_sequences=False))
  lstm_model.add(Dense(1))
  lstm_model.compile(loss='mean_squared_error', optimizer='adam')
  return(lstm_model)
}

# Function to plot results
plot_results <- function(y_test, y_pred, model_name) {
  plt.figure(figsize=(10, 6))
  plt.plot(y_test, label='True')
  plt.plot(y_pred, label=model_name)
  plt.title(paste(model_name, "'s Prediction"))
  plt.xlabel('Observation')
  plt.ylabel('ICU')
  plt.legend()
  plt.show()
}

# Main function to run analysis
run_analysis <- function(data_path) {
  # Load and preprocess data
  df = load_data(data_path)
  df = preprocess_data(df)
  
  # Split data into train and test sets
  split_date = pd.Timestamp('2021-05-01')
  list(train, test) = split_data(df['bc_icu_active'], split_date)
  
  # Scale data
  list(train_sc, test_sc) = scale_data(train, test)
  
  # Prepare data for ANN
  X_train = train_sc[:-1]
  y_train = train_sc[1:]
  X_test = test_sc[:-1]
  y_test = test_sc[1:]
  
  # Create and train ANN model
  nn_model = create_ann_model(1)
  early_stop = EarlyStopping(monitor='loss', patience=2, verbose=1)
  nn_model.fit(X_train, y_train, epochs=100, batch_size=1, verbose=1, callbacks=[early_stop], shuffle=False)
  
  # Predictions with ANN
  y_pred_test_nn = nn_model.predict(X_test)
  y_train_pred_nn = nn_model.predict(X_train)
  
  # Evaluate ANN model
  print("ANN - Train R2 score: {:0.3f}".format(r2_score(y_train, y_train_pred_nn)))
  print("ANN - Test R2 score: {:0.3f}".format(r2_score(y_test, y_pred_test_nn)))
  
  # Prepare data for LSTM
  X_train_lmse = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
  X_test_lmse = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
  
  # Create and train LSTM model
  lstm_model = create_lstm_model(X_train_lmse.shape[1])
  lstm_model.fit(X_train_lmse, y_train, epochs=100, batch_size=1, verbose=1, shuffle=False, callbacks=[early_stop])
  
  # Predictions with LSTM
  y_pred_test_lstm = lstm_model.predict(X_test_lmse)
  y_train_pred_lstm = lstm_model.predict(X_train_lmse)
  
  # Evaluate LSTM model
  print("LSTM - Train R2 score: {:0.3f}".format(r2_score(y_train, y_train_pred_lstm)))
  print("LSTM - Test R2 score: {:0.3f}".format(r2_score(y_test, y_pred_test_lstm)))
  
  # Plotting results
  plot_results(y_test, y_pred_test_nn, 'NN')
  plot_results(y_test, y_pred_test_lstm, 'LSTM')
}

# Replace with the actual path to your data file
run_analysis("path/to/your/data.csv")

