import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from nltk.sentiment.vader import SentimentIntensityAnalyzer
# read in the data
df = pd.read_csv('mental_health.csv')

# vectorize the symptom text using TF-IDF
tfidf = TfidfVectorizer()
X_transformed = tfidf.fit_transform(df['text'])

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_transformed, df['label'], test_size=0.2, random_state=42)

# train a Naive Bayes classifier on the training set
model = MultinomialNB()
model.fit(X_train, y_train)

# predict labels for the test set
y_pred = model.predict(X_test)

# calculate accuracy and other metrics
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

analyzer = SentimentIntensityAnalyzer()
df['vader_score'] = df['text'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
df['vader_sentiment'] = np.where(df['vader_score'] > 0, 'positive', 'negative')

print('Accuracy:', accuracy)
print('Confusion matrix:\n', cm)
print('Classification report:\n', report)

# compare predicted labels with true labels
df['predicted_label'] = model.predict(tfidf.transform(df['text']))
df['label_match'] = np.where(df['predicted_label']==df['label'], 'match', 'mismatch')

print(df[['text', 'label', 'predicted_label', 'label_match','vader_score','vader_sentiment']])

#bar chart for predicted labels
# read in the data
df = pd.read_csv('mental_health.csv')

# vectorize the symptom text using TF-IDF
tfidf = TfidfVectorizer()
X_transformed = tfidf.fit_transform(df['text'])

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_transformed, df['label'], test_size=0.2, random_state=42)

# train a Naive Bayes classifier on the training set
model = MultinomialNB()
model.fit(X_train, y_train)

# predict labels for the test set
y_pred = model.predict(X_test)

# compare predicted labels with true labels
df['predicted_label'] = model.predict(tfidf.transform(df['text']))
df['label_match'] = np.where(df['predicted_label']==df['label'], 'match', 'mismatch')

# create a new column for the predicted sentiment
df['predicted_sentiment'] = np.where(df['predicted_label'] > 0, 'positive', 'negative')

# create a bar chart to visualize the predicted labels as positive or negative
fig, ax = plt.subplots()
df['predicted_sentiment'].value_counts().plot(kind='bar', ax=ax)
ax.set_title('Predicted Sentiment')
ax.set_xlabel('Sentiment')
ax.set_ylabel('Count')

plt.show()
