import pandas as pd
import json
df=pd.read_csv("./data/train.csv",index_col=None)
print(json.loads(df.loc[1,'word_seq'])[0])