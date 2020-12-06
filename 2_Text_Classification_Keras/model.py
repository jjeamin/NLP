import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential, load_model


def base_model(vocab_size, embedding_matrix=None, max_len=100):
    model = Sequential()

    if embedding_matrix is None:
        model.add(Embedding(vocab_size, max_len, input_length = max_len, trainable=False)) 
    else:
        model.add(Embedding(vocab_size, max_len, weights = [embedding_matrix], input_length = max_len, trainable=False))

    model.add(SpatialDropout1D(0.3))
    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    return model