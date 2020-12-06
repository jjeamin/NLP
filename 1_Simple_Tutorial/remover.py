import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

'''
불용어 : 큰 의미가 없는 단어
'''

def remove(sentence,stop_words):
    words = word_tokenize(sentence)
    result = []

    for word in words:
        if word not in stop_words:
            result.append(word)

    return result


sentence = "Family is not an important thing. It's everything."
stop_words = stopwords.words('english')

print(remove(sentence,stop_words))
