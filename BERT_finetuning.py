import pickle
from itertools import chain

import torch
from pytorch_pretrained_bert import BertTokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm

if torch.cuda.is_available():
    device = torch.cuda.device("cuda")
else:
    device = torch.device("cpu")
print(torch.cuda.device_count())
print(torch.cuda.current_device())
print(torch.cuda.get_device_name())

# Load the data for three splits
train_dict = pickle.load(open("./data/train.pkl", 'rb'))
val_dict = pickle.load(open("./data/val.pkl", 'rb'))
test_dict = pickle.load(open("./data/test.pkl", 'rb'))

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)

# Build the tag <--> index dictionary and add PAD tag into it
tag_list = list(set(chain(*train_dict["tag_seq"])))
tag_list.append("PAD")
tag_to_index_dict = {t: i for i, t in enumerate(tag_list)}
index_to_tag_dict = {i: t for i, t in enumerate(tag_list)}


# The function to tokenize the words and extend the tags of a word to each sub word
def tokenize_words_tags(words: list, tags: list):
    tokenized_list = []
    tags_list = []
    for word, tag in zip(words, tags):
        tokenize_word = tokenizer.tokenize(word)
        number_of_sub_word = len(tokenize_word)
        tokenized_list.extend(tokenize_word)
        tags_list.extend([tag for i in range(number_of_sub_word)])
    return tokenized_list, tags_list


# Tokenize the text and tags using the function above
tokenized_text_and_tag = [tokenize_words_tags(sentence, tags) for sentence, tags in
                          zip(train_dict['word_seq'], train_dict['tag_seq'])]

# Divide the tokenized text and tag to two lists, check the max length, can not over 512
tokenized_text = [tokenized_pair[0] for tokenized_pair in tokenized_text_and_tag]
tokenized_tag = [tokenized_pair[1] for tokenized_pair in tokenized_text_and_tag]
print(
    "max length of the training tokenized text is {}".format(max([len(t_sentences) for t_sentences in tokenized_text])))
print("Length of training text and tag is {} and {}".format(len(tokenized_text), len(tokenized_tag)))


# The function to cut the over length sentence/tags into two lists
def cut_over_sized_token_list(text: list, tag: list):
    return_text, return_tag = [], []
    for index, text_list in tqdm(enumerate(text)):
        tags_list = tag[index]
        if len(text_list) > 500:
            ind = 350
            while True:
                if text_list[ind][0] == '#':
                    ind += 1
                    continue
                else:
                    text_list1, tag_list1 = text_list[:ind], tags_list[:ind]
                    text_list2, tag_list2 = text_list[ind:], tags_list[ind:]
                    return_text.append(text_list1)
                    return_text.append(text_list2)
                    return_tag.append(tag_list1)
                    return_tag.append(tag_list2)
                    break
        else:
            return_tag.append(tags_list)
            return_text.append(text_list)
    return return_text, return_tag


# Cut the over length list and check the max length again
tokenized_text, tokenized_tag = cut_over_sized_token_list(tokenized_text, tokenized_tag)

print("max length of the training tokenized text (after halved) is {}".format(
    max([len(t_sentences) for t_sentences in tokenized_text])))
print("Length of training text and tag (after halved) is {} and {}".format(len(tokenized_text), len(tokenized_tag)))
print("training words and tags tokenized are prepared\n")

# Pad all the sequences with maxlength 512 and default value 0.0 (PAD)
train_text_id_list_padded = pad_sequences(
    [tokenizer.convert_tokens_to_ids(tokenized_sentence) for tokenized_sentence in tokenized_text], maxlen=512,
    dtype="long", value=0.0, truncating="post", padding="post")
train_tags_id_list_padded = pad_sequences([[tag_to_index_dict[tag] for tag in tags] for tags in tokenized_tag],
                                          maxlen=512,
                                          dtype="int", value=tag_to_index_dict["PAD"], truncating="post",
                                          padding="post")

# Add the attention mask
train_attention_mask = [[float(value != 0.0) for value in sentence_tokens] for sentence_tokens in
                        train_text_id_list_padded]

# Do the same preprocessing for validation dataset
val_text_and_tag = [tokenize_words_tags(sentence, tags) for sentence, tags in
                    zip(val_dict['word_seq'], val_dict['tag_seq'])]
val_tokenized_text = [tokenized_pair[0] for tokenized_pair in val_text_and_tag]
val_tokenized_tag = [tokenized_pair[1] for tokenized_pair in val_text_and_tag]
print("max length of the validation tokenized text is {}".format(
    max([len(t_sentences) for t_sentences in val_tokenized_text])))
print("Length of validation text and tag is {} and {}".format(len(val_tokenized_text), len(val_tokenized_tag)))
val_tokenized_text, val_tokenized_tag = cut_over_sized_token_list(val_tokenized_text, val_tokenized_tag)
print("max length of the validation tokenized text (after halved) is {}".format(
    max([len(t_sentences) for t_sentences in val_tokenized_text])))
print("Length of validation text and tag (after halved) is {} and {}".format(len(val_tokenized_text),
                                                                             len(val_tokenized_tag)))
print("validation words and tags tokenized are prepared\n")
val_text_id_list_padded = pad_sequences(
    [tokenizer.convert_tokens_to_ids(tokenized_sentence) for tokenized_sentence in val_tokenized_text], maxlen=512,
    dtype="long", value=0.0, truncating="post", padding="post")
val_tags_id_list_padded = pad_sequences([[tag_to_index_dict[tag] for tag in tags] for tags in val_tokenized_tag],
                                        maxlen=512,
                                        dtype="int", value=tag_to_index_dict["PAD"], truncating="post", padding="post")
val_attention_mask = [[float(value != 0.0) for value in sentence_tokens] for sentence_tokens in val_text_id_list_padded]

# We are all set, transform everything into torch tensor.
train_input = torch.tensor(train_text_id_list_padded)
train_tag = torch.tensor(train_tags_id_list_padded)
train_mask = torch.tensor(train_attention_mask)
val_input = torch.tensor(val_text_id_list_padded)
val_tag = torch.tensor(val_tags_id_list_padded)
val_mask = torch.tensor(val_attention_mask)
