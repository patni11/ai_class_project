import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Conv1D, MaxPooling1D, Flatten, TimeDistributed
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.regularizers import l2
import numpy as np

text_df = pd.read_csv('text_data.csv')
financial_data = pd.read_csv('btc_price_data.csv')
time_steps = 10
financial_data['date'] = pd.to_datetime(financial_data['date'], format='%Y-%m-%d %H:%M:%S')
text_df['date'] = pd.to_datetime(text_df['date'], format='%m/%d/%y')
text_df['next_day'] = text_df['date'] + pd.Timedelta(days=1)

# Align financial data with the next day's text sentiment data
df_price = pd.merge_asof(financial_data.sort_values('date'), text_df.sort_values('next_day'), left_on='date', right_on='next_day', direction='forward')

#df_price.to_csv('df_price.csv', index=False)

def create_sequences(X, y, time_steps):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X.iloc[i:(i + time_steps)].values)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

features = ['neg', 'neu', 'compound', 'pos', "open", "Volume USD", "RSI", "MACD", "Signal_Line", "Upper_Band", "Lower_Band", "SMA", "EMA"]
X = df_price[features]
y = df_price['close']
X_seq, y_seq = create_sequences(X, y, time_steps)

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train_seq, X_test_seq, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2)

# Normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)
X_train_scaled_seq = np.array([scaler.fit_transform(x) for x in X_train_seq])
X_test_scaled_seq = np.array([scaler.transform(x) for x in X_test_seq])

# Reshape for LSTM [samples, time steps, features]
# X_train_reshaped = np.reshape(X_train_scaled, (X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
# X_test_reshaped = np.reshape(X_test_scaled, (X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

X_train_reshaped = X_train_scaled_seq
X_test_reshaped = X_test_scaled_seq

# Define the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))  # Last LSTM layer
model.add(Dropout(0.2))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train_reshaped, y_train, epochs=10, batch_size=32, validation_data=(X_test_reshaped, y_test), verbose=1)
model.save('my_model.h5')
# predictions
#predictions = model.predict(X_test_reshaped)

predictions = model.predict(X_test_reshaped).flatten() 

# metrics
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R^2 Score: {r2}")


#PLOTTING
plt.rcParams['agg.path.chunksize'] = 10000  # Adjust the value as needed, start with 10000
plt.rcParams['path.simplify_threshold'] = 0.1  # Adjust as needed

# plt.figure(figsize=(10, 6))
# plt.plot(y_test.reset_index(drop=True), label='Actual Price')  # Reset index to ensure proper plotting
# plt.plot(predictions, label='Predicted Price')
# plt.title('Price Prediction')
# plt.xlabel('Time')
# plt.ylabel('Price')
# plt.legend()
# plt.ylim([50000, max(y_test.max(), predictions.max()) + 1000])  # Set the y-axis to start at 50,000 and go up to a little above the max value in your data
# plt.show()

plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Actual Price')  # y_test is already a 1D numpy array
plt.plot(predictions, label='Predicted Price')
plt.title('Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.show()