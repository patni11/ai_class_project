import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import numpy as np

text_df = pd.read_csv('text_data.csv')
financial_data = pd.read_csv('btc_price_data.csv')

financial_data['date'] = pd.to_datetime(financial_data['date'], format='%Y-%m-%d %H:%M:%S')
text_df['date'] = pd.to_datetime(text_df['date'], format='%m/%d/%y')
text_df['next_day'] = text_df['date'] + pd.Timedelta(days=1)

# Align financial data with the next day's text sentiment data
df_price = pd.merge_asof(financial_data.sort_values('date'), text_df.sort_values('next_day'), left_on='date', right_on='next_day', direction='forward')
#print(df_price.head())
# Print the first and last dates from the merged DataFrame
# print("First Date:", df_price['date_x'].iloc[0])
# print("Last Date:", df_price.iloc[-1])
# print("Number of rows in the financial data:", df_price.shape[0])

features = ['neg', 'neu', 'pos', 'compound', "open", "Volume USD", "RSI", "MACD", "Signal_Line", "Upper_Band", "Lower_Band", "SMA", "EMA"]
X = df_price[features]
y = df_price['close']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape input to be [samples, time steps, features]
# This step is crucial and depends on your data. You need to reshape it into a 3D array
X_train_reshaped = np.reshape(X_train_scaled, (X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_reshaped = np.reshape(X_test_scaled, (X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# Define the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=50,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train_reshaped, y_train, epochs=200, batch_size=32, validation_data=(X_test_reshaped, y_test), verbose=1)


# predictions
predictions = model.predict(X_test_reshaped)


# metrics
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R^2 Score: {r2}")

# Previous Values
# Mean Squared Error (MSE): 157173799.44795886
# Mean Absolute Error (MAE): 7099.468502949072
# R^2 Score: 0.45268490520174076

#PLOTTING
plt.rcParams['agg.path.chunksize'] = 10000  # Adjust the value as needed, start with 10000
plt.rcParams['path.simplify_threshold'] = 0.1  # Adjust as needed


plt.figure(figsize=(10, 6))
plt.plot(y_test.reset_index(drop=True), label='Actual Price')  # Reset index to ensure proper plotting
plt.plot(predictions, label='Predicted Price')
plt.title('Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()