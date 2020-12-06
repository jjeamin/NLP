import nltk,konlpy
from nltk.tokenize import word_tokenize
from nltk.tokenize import WordPunctTokenizer
from nltk.tokenize import TreebankWordTokenizer
from nltk.tokenize import sent_tokenize
from konlpy.tag import Okt

# apostrophe = '
def token(text,mode):
    if mode == 'sequence':
        token_list = sent_tokenize(text)
    elif mode == 'apostrophe':
        token_list = WordPunctTokenizer().tokenize(text)
    elif mode == 'hyphen':
        token_list = TreebankWordTokenizer().tokenize(text)
    else:
        token_list = word_tokenize(text)

    return token_list

def ko_token(text):
    return Okt().morphs(text)

'''
PRP는 인칭 대명사, VBP는 동사, 
RB는 부사, VBG는 현재부사, 
IN은 전치사, NNP는 고유 명사, 
NNS는 복수형 명사, CC는 접속사, DT는 관사
'''
def tag(token_list):
    return nltk.tag.pos_tag(token_list)

def ko_tag(text):
    return Okt().pos(token_list)


ko = "열심히 코딩한 당신, 연휴에는 여행을 가봐요"
token_list = ko_token(ko)
tags = tag(ko,lang='ko')
print(token_list)
print(tags)
