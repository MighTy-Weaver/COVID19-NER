import pickle

from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
test = pickle.load(open('./data/train.pkl', 'rb'))
for i in range(len(test['word_seq'])):
    print(len(tokenizer.encode(test['word_seq'][i])))

a = [1, 2, 3, 4, 5]
print(a[1:-1])
