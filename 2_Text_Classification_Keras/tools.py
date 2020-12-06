import pickle
import re
import ast
import numpy as np
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def save_pkl(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def load_pkl(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)  # 단 한줄씩 읽어옴

    return data


def get_stopwords():
    stopwords = load_pkl('./stopwords.pkl')

    return stopwords


def text_preprocessing(text_list):
    stopwords = get_stopwords()
    tokenizer = Okt() #형태소 분석기 
    token_list = []
    
    for text in text_list:
        txt = re.sub('[^가-힣a-z]', ' ', text) #한글과 영어 소문자만 남기고 다른 글자 모두 제거
        token = tokenizer.morphs(txt) #형태소 분석
        token = [t for t in token if t not in stopwords or type(t) != float] #형태소 분석 결과 중 stopwords에 해당하지 않는 것만 추출
        token_list.append(token)
        
    return token_list, tokenizer


def text2sequence(text, max_len=100):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text)
    train_X_seq = tokenizer.texts_to_sequences(text)
    vocab_size = len(tokenizer.word_index) + 1
    print('vocab_size : ', vocab_size)
    X_train = pad_sequences(train_X_seq, maxlen = max_len)
    
    return X_train, vocab_size, tokenizer


def str_to_list(x):
    x = ast.literal_eval(x)
    x = [n.strip() for n in x]

    return x


def glove_word2vec(path):
    word2vec = dict()
    f = open(path, 'r', encoding='UTF8')
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype='float32')
        word2vec[word] = vector
    f.close()

    return word2vec