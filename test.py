import json
import pickle

import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("./data/train.csv", index_col=None)
print(json.loads(df.loc[1, 'word_seq'])[0])

data = pickle.load(open('./data/train.pkl', 'rb'))
print(max([len(data['word_seq'][i]) for i in range(len(data))]))

a = [[1], [2], [3]]
a.append([4])
print(a)

plt.plot([1,2,3],[1,2,3])
plt.show()
