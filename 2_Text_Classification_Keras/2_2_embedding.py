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

    model = gensim.models.Word2Vec(sentences = x_data, size = 100, window = 5, min_count = 0, workers = 1, sg = 0)
    return model


if __name__ == "__main__":
    DATA_PATH = Path("data")
    csv_path = DATA_PATH / "prep_news_train.csv"
    embedding_path = DATA_PATH / 'news.embedding'
    train = False

    if train:
        model = embedding(csv_path)
        model.save()
    else:
        model = gensim.models.Word2Vec.load(embedding_path)
        print(model.wv.most_similar("사람"))