import pandas as pd
import numpy as np

class MarketIndicators:
    def __init__(self, df):
        self.df = df

    def RSI(self, period=14):
        delta = self.df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        RS = gain / loss
        return 100 - (100 / (1 + RS))

    def MACD(self, slow=26, fast=12, signal=9):
        exp1 = self.df['close'].ewm(span=fast, adjust=False).mean()
        exp2 = self.df['close'].ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd, signal_line

    def BollingerBands(self, period=20, num_of_std=2):
        sma = self.df['close'].rolling(window=period).mean()
        std = self.df['close'].rolling(window=period).std()
        upper_band = sma + (std * num_of_std)
        lower_band = sma - (std * num_of_std)
        return upper_band, lower_band

    def SMA(self, period=20):
        return self.df['close'].rolling(window=period).mean()

    def EMA(self, period=20):
        return self.df['close'].ewm(span=period, adjust=False).mean()



file_path = 'ordered_price_data.csv'
df = pd.read_csv(file_path)
df['date'] = pd.to_datetime(df['date'])

market_indicators = MarketIndicators(df)

# Calculating each indicator
df['RSI'] = market_indicators.RSI().fillna(0)
macd, signal_line = market_indicators.MACD()
df['MACD'] = macd.fillna(0)
df['Signal_Line'] = signal_line.fillna(0)
upper_band, lower_band = market_indicators.BollingerBands()
df['Upper_Band'] = upper_band.fillna(0)
df['Lower_Band'] = lower_band.fillna(0)
df['SMA'] = market_indicators.SMA().fillna(0)
df['EMA'] = market_indicators.EMA().fillna(0)

#First 10 columns
print(df.tail(10))
df.to_csv('btc_price_data.csv', index=False)
