import pandas as pd
import nltk
from nltk.corpus import stopwords
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

# 讀取Sentiment140資料集
df = pd.read_csv('training.1600000.processed.noemoticon.csv', encoding='latin-1', header=None)
df.columns = ['target', 'id', 'date', 'flag', 'user', 'text']
df = df[['target', 'text']]
df['target'] = df['target'].replace({4: 1})

# 清理文本數據
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    text = ' '.join([stemmer.stem(word) for word in text.split()])
    return text

df['text'] = df['text'].apply(clean_text)

# 拆分數據集為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['target'], test_size=0.2, random_state=42)

# 使用TF-IDF進行特徵提取
nltk.download('stopwords')
vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'), max_features=5000,ngram_range=(1,2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 訓練Logistic Regression模型
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# 預測測試集
y_pred = model.predict(X_test_tfidf)

# 評估模型
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(classification_report(y_test, y_pred))

# 使用模型進行新句子情緒預測
def predict_sentiment(text):
    text_cleaned = clean_text(text)
    text_tfidf = vectorizer.transform([text_cleaned])
    prediction = model.predict(text_tfidf)
    return 'Negative' if prediction == 0 else 'Positive'

new_sentence = "Shuts up"
print(f'The sentiment of the sentence "{new_sentence}" is: {predict_sentiment(new_sentence)}')
