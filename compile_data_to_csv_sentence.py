import json
import pickle

from pandas import DataFrame

# This code compiles the three split dataset into json dumped csv and
# concat the words into a sentence, store it in another csv file.

for split in ["val", "test", "train"]:
    df = DataFrame(columns=['id', 'word_seq', 'tag_seq'])
    df_concat = DataFrame(columns=['id', 'word_seq', 'tag_seq'])
    data = pickle.load(open('./data/{}.pkl'.format(split), 'rb'))
    df['id'] = df_concat['id'] = data['id']
    for i in range(len(df)):
        df.loc[i, 'word_seq'] = json.dumps(data['word_seq'][i])
        df_concat.loc[i, 'word_seq'] = ' '.join(data['word_seq'][i])
        try:
            df.loc[i, 'tag_seq'] = json.dumps(data['tag_seq'][i])
            df_concat.loc[i, 'tag_seq'] = json.dumps(data['tag_seq'][i])
        except KeyError:
            pass
    df.to_csv("./data/{}.csv".format(split), index=False)
    df_concat.to_csv("./data/{}_word_concat.csv".format(split), index=False)
