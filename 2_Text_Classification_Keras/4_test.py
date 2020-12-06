import pandas as pd
import numpy as np
import gensim
from model import base_model
from pathlib import Path
from tools import str_to_list, text2sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences

DATA_PATH = Path("data")

max_len = 100
vocab_size = 43500

train = pd.read_csv(DATA_PATH / "prep_news_train.csv")
test = pd.read_csv(DATA_PATH / "prep_news_test.csv")
submission = pd.read_csv(DATA_PATH / "sample_submission.csv")

train_data = [ str_to_list(sentence) for sentence in train.new_article.values ]
test_data = [ str_to_list(sentence) for sentence in test.new_article.values ]

_, _, tokenizer = text2sequence(train_data, max_len = max_len)

test_X_seq = tokenizer.texts_to_sequences(test_data)
test_X = pad_sequences(test_X_seq, maxlen = max_len)

model = base_model(vocab_size=vocab_size)
model.load_weights('./glove_model.h5')

predictions = model.predict(test_X)
print(predictions)
predicted_label = np.where(predictions > 0.5, 1, 0)
print(predicted_label)

submission['info'] = predicted_label.reshape(-1)

submission.to_csv('glove_model.csv', index = False)