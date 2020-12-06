import os
import pandas as pd
from tools import *
from pathlib import Path

DATA_PATH = Path("data")

train = pd.read_csv(DATA_PATH / "news_train.csv")
test = pd.read_csv(DATA_PATH / "news_test.csv")

train['new_article'], okt = text_preprocessing(train['content'])
test['new_article'], okt = text_preprocessing(test['content'])

train.to_csv(DATA_PATH / "prep_news_train.csv", index=False)
test.to_csv(DATA_PATH / "prep_news_test.csv", index=False)