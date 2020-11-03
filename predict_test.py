import pickle

from pandas import DataFrame
from tqdm import trange

test_data = pickle.load(open("./data/test.pkl", 'rb'))
answer = DataFrame(columns=['id', 'labels'])
answer['id'] = test_data['id']
print(len(test_data['word_seq']))
# for i in trange(len(test_data)):
#     sentence = ' '.join(test_data['word_seq'][i])
#     print(sentence)