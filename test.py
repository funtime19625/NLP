import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import subprocess

nltk.download('punkt')
nltk.download('stopwords')
def load_inappropriate_words(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return set(line.strip().lower() for line in file)
    
def calculate_inappropriate_score(text, inappropriate_words):
    # 分詞
    tokens = word_tokenize(text.lower())
    # 停用詞列表
    stop_words = set(stopwords.words('english'))
    # 過濾停用詞
    tokens = [word for word in tokens if word not in stop_words]
    # 計算文本中不當字眼的數量
    print(inappropriate_words)
    print(tokens)
    inappropriate_count = sum(1 for word in tokens if word in inappropriate_words)
    # 計算不當字眼比例
    inappropriate_score = inappropriate_count / len(tokens)
    return inappropriate_score

# 不當字眼列表
inappropriate_words = load_inappropriate_words('DangerousWords.txt')

# 測試文本
test_text = "I hate this Grape. It's terrible!"

# 計算不當字眼分數
score = calculate_inappropriate_score(test_text, inappropriate_words)
print("Inappropriate score:", score)
if score > 0.01:
    print("This message contains attack words.")
    subprocess.run(['shutdown', '/r', '/t', '1'])
elif score < 0.01 and score >=0.0001:
    print("This message contains sensitive information.")
else:
    print("This message does not contain attack words.")