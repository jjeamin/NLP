import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm

DATA_PATH = Path("data")

train = pd.read_csv(DATA_PATH / "news_train.csv")
test = pd.read_csv(DATA_PATH / "news_test.csv")
submission = pd.read_csv(DATA_PATH / "sample_submission.csv")

print(train['date'].min())
print(test['date'].min())
print(train['date'].max())
print(test['date'].max())

train_unique_ad_sentence = train.query('info == "1"')['content'].unique()
test_unique_sentence = test['content'].unique()

print(len(train_unique_ad_sentence))
print(len(test_unique_sentence))
print(len(set(train_unique_ad_sentence) & set(test_unique_sentence)))

test_content = test['content'].values

for idx, sent in enumerate(tqdm(test_content)) : #Test 데이터에 있는 모든 content들에 대하여

    if sent in train_unique_ad_sentence: # Train 데이터의 광고성 문구와 같은지 비교
        submission['info'].iloc[idx] = 1 # 같으면 1

    else : 
        submission['info'].iloc[idx] = 0 # 다르면 0

print(submission.head())

submission.to_csv('rule_based.csv', index = False)

# score : 95.62