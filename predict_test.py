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

# load the training dict and build the tag <--> index dictionary and add PAD tag into it
train_dict = pickle.load(open("./data/train.pkl", 'rb'))
tag_list = list(set(chain(*train_dict["tag_seq"])))
tag_list.append("PAD")
tag_to_index_dict = {t: i for i, t in enumerate(tag_list)}
index_to_tag_dict = {i: t for i, t in enumerate(tag_list)}

# Build an arg_parser to retrieve the model path
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", required=True, type=str, help="The PATH of the model")
args = parser.parse_args()
parameters = args.model_path.split('/')[-1].replace("model_", '').replace(".pkl", '')
# Load the test dataset and construct the Dataframe
test_data = pickle.load(open("./data/test.pkl", 'rb'))
answer = DataFrame(columns=['id', 'labels'])
answer['id'] = test_data['id']

# Set up the same tokenizer and load the model, push the model to GPU and eval mode
tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
model = torch.load(args.model_path)
model.eval()
model.cuda()

# Looping through all the sentences
for i in trange(len(test_data['word_seq'])):
    decoded_tokens, tag_predicted = None, None

    # Tokenize the sentence and push the encoded text to GPU
    sentence = ' '.join(test_data['word_seq'][i])
    tokenized_text = tokenizer.tokenize(sentence)

    if len(tokenized_text) < 512:
        # push the text into GPU, get the output of the prediction, push it to CPU
        tokenized_index = tokenizer.convert_tokens_to_ids(tokenized_text)
        input_text = torch.tensor([tokenized_index]).cuda()
        with torch.no_grad():
            output = model(input_text)
        tag_predicted = np.argmax(output[0].to('cpu').numpy(), axis=2)[0]
        decoded_tokens = tokenizer.convert_ids_to_tokens(input_text.to('cpu').numpy()[0])
        print(decoded_tokens)
        print(tag_predicted)
        print(len(decoded_tokens),len(tag_predicted))
    else:
        ind = 350
        while True:
            if tokenized_text[ind].startswith("##"):
                ind += 1
                continue
            else:
                text_list1, text_list2 = tokenized_text[:ind], tokenized_text[ind:]
                index_list1, index_list2 = tokenizer.convert_tokens_to_ids(text_list1), tokenizer.convert_tokens_to_ids(
                    text_list2)
                input_index1, input_index2 = torch.tensor([index_list1]).cuda(), torch.tensor([index_list2]).cuda()
                with torch.no_grad():
                    output1 = model(input_index1)
                    output2 = model(input_index2)
                tag_predicted1 = np.argmax(output1[0].to('cpu').numpy(), axis=2)[0]
                tag_predicted2 = np.argmax(output2[0].to('cpu').numpy(), axis=2)[0]
                decoded_tokens = tokenizer.convert_ids_to_tokens(index_list1[:-1] + index_list2[1:])
                tag_predicted = []
                tag_predicted.extend(tag_predicted1)
                tag_predicted.extend(tag_predicted2)
                break

    # Merge the divided tokens and tags together
    rebuilt_tokens, rebuilt_tags = [], []
    for token, tag in zip(decoded_tokens, tag_predicted):
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
answer.to_csv("./answers/answer_{}.csv".format(parameters), index=True)
