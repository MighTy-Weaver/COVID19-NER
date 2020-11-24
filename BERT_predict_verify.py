import argparse
import json
import os
import pickle

import numpy as np
import torch
from pandas import DataFrame
from tqdm import trange
from transformers import BertTokenizer

# Build an arg_parser to retrieve the model path
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", required=True, type=str, help="The PATH of the model")
parser.add_argument("--print", required=False, type=bool, default=False, help="Print the difference or not")
parser.add_argument("--split", required=False, type=str, default="test", choices=['test', 'val', 'train'],
                    help="Which dataset you want to test on")
args = parser.parse_args()
parameters = args.model_path.split('/')[-1].replace("model_", '').replace(".pkl", '')
e_bs = '_'.join(parameters.split('_')[:2])

# load the tag <--> index dict
tag_to_index_dict = np.load("./models/model_tag2id_{}.npy".format(e_bs), allow_pickle=True).item()
index_to_tag_dict = np.load("./models/model_id2tag_{}.npy".format(e_bs), allow_pickle=True).item()

# Load the test dataset and construct the Dataframe
test_data = pickle.load(open("./data/{}.pkl".format(args.split), 'rb'))
answer = DataFrame(columns=['id', 'labels'])
answer['id'] = test_data['id']

# Set up the same tokenizer and load the model, push the model to GPU and eval mode
tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
model = torch.load(args.model_path)
model.eval()
model.cuda()

# Looping through all the sentences
for i in trange(len(test_data['word_seq'])):
    decoded_tokens, tag_predicted = [], []

    # push the text into GPU, get the output of the prediction, push it to CPU
    tokenized_index = tokenizer.encode(test_data['word_seq'][i])
    input_text = torch.tensor([tokenized_index]).cuda()
    with torch.no_grad():
        output = model(input_text)
    tag_predicted = np.argmax(output[0].to('cpu').numpy(), axis=2)[0][1:-1]
    decoded_tokens = tokenizer.convert_ids_to_tokens(input_text.to('cpu').numpy()[0])

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
answer.to_csv("./answers/{}_answer_{}.csv".format(args.split, parameters), index=True)

# Verify the length of the generation
count = 0
for i in trange(len(answer)):
    if len(json.loads(answer.loc[i, 'labels'])) != len(test_data['word_seq'][i]):
        if args.print:
            print(len(json.loads(answer.loc[i, 'labels'])), len(test_data['word_seq'][i]))
        count += 1
print("There are in total {}/{} answers with different length".format(count, len(test_data['word_seq'])))
if count == 0:
    print("Great! Generations are of same length!")

# Calculate the accuracy for test and val
if args.split == 'val' or args.split == 'train':
    correct = 0
    total = 0
    for i in trange(len(test_data)):
        for j in range(len(test_data['tag_seq'][i])):
            if json.loads(answer.loc[i, 'labels'])[j] == test_data['tag_seq'][i][j]:
                correct += 1
            total += 1
    print("The accuracy for {} is {}".format(args.split, correct / total))
