import pandas as pd
import json
import pickle
df=pd.read_csv("./data/train.csv",index_col=None)
print(json.loads(df.loc[1,'word_seq'])[0])

data=pickle.load(open('./data/train.pkl','rb'))
print(max([len(data['word_seq'][i]) for i in range(len(data))]))