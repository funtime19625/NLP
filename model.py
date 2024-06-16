import nltk
from nltk.tokenize import word_tokenize
import subprocess
# 載入NLTK的停用詞和標點符號列表
nltk.download('stopwords')
nltk.download('punkt')

def load_words(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return set(line.strip().lower() for line in file)

def contains_attack(text, attack_words):
    # 將文本拆分成單詞
    words = word_tokenize(text.lower())
    # 檢查是否包含攻擊性詞彙
    for word in words:
        if word in attack_words:
            return True
    return False
def contains_sensitive(text, sensitive_words):
    # 檢查是否包含敏感性詞彙
    for word in text.split():
        if word.lower() in sensitive_words:
            return True
    return False

# 載入攻擊性詞彙列表
attack_words = load_words('DangerousWords.txt')
sensitive_words = load_words('SensitiveWords.txt')
# 測試文本
while(1):
    test_text = input()
    if contains_attack(test_text, attack_words):
        print("This message contains attack words.")
        subprocess.run(['shutdown', '/r', '/t', '1'])
    elif contains_sensitive(test_text, sensitive_words):
        print("This message contains sensitive information.")
    else:
        print("This message does not contain attack words.")
