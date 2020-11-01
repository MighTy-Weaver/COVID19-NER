import pickle
from itertools import chain

import torch
from pytorch_pretrained_bert import BertTokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

if torch.cuda.is_available():
    device = torch.cuda.device("cuda")
else:
    device = torch.device("cpu")
print(torch.cuda.device_count())
print(torch.cuda.current_device())
print(torch.cuda.get_device_name())

train_dict = pickle.load(open("./data/train.pkl", 'rb'))
val_dict = pickle.load(open("./data/val.pkl", 'rb'))
test_dict = pickle.load(open("./data/test.pkl", 'rb'))

tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)

tag_list = list(set(chain(*train_dict["tag_seq"])))
tag_list.append("PAD")
tag_to_index_dict = {t: i for i, t in enumerate(tag_list)}
index_to_tag_dict = {i: t for i, t in enumerate(tag_list)}


def tokenize_words_tags(words: list, tags: list):
    tokenized_list = []
    tags_list = []
    for word, tag in zip(words, tags):
        tokenize_word = tokenizer.tokenize(word)
        number_of_subwords = len(tokenize_word)
        tokenized_list.extend(tokenize_word)
        tags_list.extend(tag * number_of_subwords)
    return tokenized_list, tags_list


tokenized_text_and_tag = [tokenize_words_tags(sentence, tags) for sentence, tags in
                          zip(train_dict['word_seq'], train_dict['tag_seq'])]
print(tokenized_text_and_tag[0])
print("words and tags tokenized and paired")

tokenized_text = [tokenized_pair[0] for tokenized_pair in tokenized_text_and_tag]
tokenized_tag = [tokenized_pair[1] for tokenized_pair in tokenized_text_and_tag]

text_id_list_padded = pad_sequences(
    [tokenizer.convert_tokens_to_ids(tokenized_sentence) for tokenized_sentence in tokenized_text], maxlen=135,
    dtype="long", value=0.0, truncating="post", padding="post")
tags_id_list_padded = pad_sequences([[tag_to_index_dict[tag] for tag in tags] for tags in tokenized_tag], maxlen=135,
                                    dtype="long", value=tag_to_index_dict["PAD"], truncating="post", padding="post")
