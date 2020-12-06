import pandas as pd
import numpy as np
import gensim
import os
from model import base_model
from pathlib import Path
from tools import str_to_list, text2sequence, glove_word2vec
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger

DATA_PATH = Path("data")

max_len = 100
epochs = 10
batch_size = 128

train = pd.read_csv(DATA_PATH / "prep_news_train.csv")

word2vec = gensim.models.Word2Vec.load('./data/news_min0.embedding')
train_data = [ str_to_list(sentence) for sentence in train.new_article.values ]

train_X, vocab_size, tokenizer = text2sequence(train_data, max_len = max_len)
train_y = train['info']

word_index = tokenizer.word_index

embedding_matrix = np.zeros((vocab_size, max_len))

for word, index in word_index.items():
    if word in word2vec:
        embedding_vector = word2vec[word] 
        embedding_matrix[index] = embedding_vector 
    else:
        print("word2vec에 없는 단어입니다.")
        break

model = base_model(vocab_size=vocab_size, embedding_matrix=embedding_matrix, max_len=max_len)
model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['acc'])
model.summary()

callbacks = [
    ModelCheckpoint('models/cosine_model.h5', verbose=1, save_best_only=True),
    CSVLogger('models/cosine_log.csv'),
]

history = model.fit(train_X, train_y,
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=callbacks)

model.save_weights('model.h5')