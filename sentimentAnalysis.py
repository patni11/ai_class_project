import pandas as pd
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
import nltk

# nltk.download('vader_lexicon')
# sid = SentimentIntensityAnalyzer()

df = pd.read_csv('text_data.csv')

# def get_sentiment_score(text):
#     text = str(text)
#     return sid.polarity_scores(text)  

# df['sentiment_scores'] = df['text'].apply(get_sentiment_score)

# df_sentiment = pd.json_normalize(df['sentiment_scores'])
# df = pd.concat([df.drop(['sentiment_scores'], axis=1), df_sentiment], axis=1)
df['date'] = pd.to_datetime(df['date'])
df['date'] = df['date'].dt.strftime('%m/%d/%y')

df.to_csv('text_data.csv', index=False)
