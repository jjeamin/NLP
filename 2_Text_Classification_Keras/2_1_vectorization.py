import pandas as pd
import numpy as np
from pathlib import Path
from tools import str_to_list, text2sequence

DATA_PATH = Path("data")

train = pd.read_csv(DATA_PATH / "prep_news_train.csv")

x_data = []

for sentence in train.new_article.values:
    data = str_to_list(sentence)
    x_data.append(data)

train_X, vocab_size, tokenizer = text2sequence(x_data, max_len = 100)
train_y = train['info']

word_index = tokenizer.word_index

for i in word_index.items():
    print(i)
    break
