import argparse
import json
import os
import pickle
from itertools import chain

import numpy as np
import torch
from pandas import DataFrame
from tqdm import trange
from transformers import BertTokenizer

from BERT_finetuning import train_dict

# Build an arg_parser to retrieve the model path
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", required=True, type=str, help="The PATH of the model")
args = parser.parse_args()

# Load the test dataset and construct the Dataframe
test_data = pickle.load(open("./data/test.pkl", 'rb'))
answer = DataFrame(columns=['id', 'labels'])
answer['id'] = test_data['id']

# Build the tag <--> index dictionary and add PAD tag into it
tag_list = list(set(chain(*train_dict["tag_seq"])))
tag_list.append("PAD")
tag_to_index_dict = {t: i for i, t in enumerate(tag_list)}
index_to_tag_dict = {i: t for i, t in enumerate(tag_list)}

# Set up the same tokenizer and load the model, push the model to GPU and eval mode
tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
model = torch.load(args.model_path)
model.eval()
model.cuda()

# Looping through all the sentences
for i in trange(len(test_data['word_seq'])):
    # Tokenize the sentence and push the encoded text to GPU
    sentence = ' '.join(test_data['word_seq'][i])
    tokenized_list = tokenizer.encode(sentence)
    input_text = torch.tensor([tokenized_list]).cuda()

    # Get the output of the prediction, push it to CPU
    with torch.no_grad():
        output = model(input_text)
    tag_predicted = np.argmax(output[0].to('cpu').numpy(), axis=2)
    decoded_tokens = tokenizer.convert_ids_to_tokens(input_text.to('cpu').numpy()[0])

    # Merge the divided tokens and tags together
    rebuilt_tokens, rebuilt_tags = [], []
    for token, tag in zip(decoded_tokens, tag_predicted[0]):
        if token.startswith("##"):
            rebuilt_tokens[-1] = rebuilt_tokens[-1] + token[2:]
        else:
            rebuilt_tokens.append(token)
            rebuilt_tags.append(index_to_tag_dict[tag])

    # Use json to dump the result and save it into the dataframe
    dumped = json.dumps(rebuilt_tags)
    answer.loc[i, 'labels'] = dumped

# Save the dataframe as a csv
if not os.path.exists("./answers/"):
    os.mkdir("./answers/")
answer.to_csv("./answers/answer.csv", index=True)
