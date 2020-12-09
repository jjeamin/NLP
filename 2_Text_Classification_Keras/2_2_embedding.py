import re
import gensim
import pandas as pd
from konlpy.tag import Okt
from pathlib import Path
from tools import str_to_list

def embedding(csv_path):
    train = pd.read_csv(csv_path)

    x_data = []

    for sentence in train.new_article.values:
        data = str_to_list(sentence)
        x_data.append(data)

    w2v_model = gensim.models.word2vec.Word2Vec(size = 100, window = 5, min_count = 2)

    w2v_model.build_vocab(x_data)
    words = w2v_model.wv.vocab.keys()
    vocab_size = len(words)
    print("Vocab size", vocab_size)

    # Train Word Embeddings
    w2v_model.train(x_data, total_examples=len(x_data), epochs=100)

    return w2v_model


if __name__ == "__main__":
    DATA_PATH = Path("data")
    csv_path = DATA_PATH / "prep_news_train.csv"
    embedding_path = DATA_PATH / 'news_embedding.txt'
    train = True

    if train:
        w2v_model = embedding(csv_path)
        w2v_model.save(embedding_path)
    else:
        w2v_model = gensim.models.Word2Vec.load(embedding_path)
        print(w2v_model.wv.most_similar("사람"))