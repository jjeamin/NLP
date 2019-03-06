import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

'''
표제어 추출 : Lemmatization
1) stem(어간) : 단어의 의미를 담고있는 단어의 핵심 부분
2) affix(접사) : 단어에 추가적인 의미를 주는 부분
cats : cat(stem) + s(affix)

am -> be
the going -> the going
having -> have
'''

def lemma(word,part):
    n = WordNetLemmatizer()

    return n.lemmatize(word,part)

'''
어간 추출 : Stemming

am -> am
the going -> the go
having -> hav
'''
def stem(word):
    s = PorterStemmer()

    return s.stem(word)


print(lemma('am','v'))
print(stem('formalize'))

'''
불용어 : Stopword
'''