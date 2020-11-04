import pickle
from itertools import chain
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizer


# load the training dict and build the tag <--> index dictionary and add PAD tag into it
train_dict = pickle.load(open("./data/train.pkl", 'rb'))
tag_list = list(set(chain(*train_dict["tag_seq"])))
tag_list.append("PAD")
tag_to_index_dict = {t: i for i, t in enumerate(tag_list)}
index_to_tag_dict = {i: t for i, t in enumerate(tag_list)}
print(index_to_tag_dict)

data = pickle.load(open('./data/test.pkl', 'rb'))
answer = pd.read_csv("F:/大三上/COMP4901K/answer_e5_bs2_epoch4.csv", index_col=0)
import json
line=0
ans_list=json.loads(answer.loc[line, 'labels'])

print(len(ans_list))
print(len(data['word_seq'][line]))
