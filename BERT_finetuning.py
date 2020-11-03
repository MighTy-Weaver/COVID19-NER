import argparse
import os
import pickle
from itertools import chain

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from pytorch_pretrained_bert import BertTokenizer
from sklearn.metrics import accuracy_score, f1_score
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler, DataLoader
from tqdm import tqdm, trange
from transformers import BertForTokenClassification, AdamW, get_linear_schedule_with_warmup

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, required=False, default=10,
                    help="The number of epochs to be train for the model")
parser.add_argument("--gradnorm", type=float, required=False, default=1.0, help="maximum gradient normalization")
args = parser.parse_args()

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
        tags_list.extend([tag for _ in range(number_of_sub_word)])
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
print("\nTensors Created\n")

# Make data loaders and make batch
train_dataset = TensorDataset(train_input, train_mask, train_tag)
val_dataset = TensorDataset(val_input, val_mask, val_tag)
train_sample = RandomSampler(train_dataset)
val_sample = SequentialSampler(val_dataset)
train_loader = DataLoader(train_dataset, sampler=train_sample, batch_size=32)
val_loader = DataLoader(val_dataset, sampler=val_sample, batch_size=32)

# Define our model
model = BertForTokenClassification.from_pretrained("bert-base-cased", num_labels=len(tag_to_index_dict),
                                                   output_attentions=False, output_hidden_states=False)

# push the model to GPU
model.cuda()

# Setup our parameter training optimizer and pass the parameters
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}]
optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5, eps=1e-8)

# Set the training epochs and grad_norm, calculate the total steps of training
epochs = args.epoch
max_grad_norm = args.gradnorm
total_steps = len(train_loader) * epochs

# Create the learning rate scheduler, training steps is total step
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# ################## We are ready to start the fine tuning     ###################
# ################## The training and validation starts here   ###################

print("Start Training.......")
# Store the loss and with two list for each training and validation
training_loss_values, validation_loss_values = [], []

for _ in trange(epochs, desc="Epoch: "):
    model.train()
    total_loss = 0
    for step, batch in enumerate(train_loader):
        batch_data = tuple(t.to(device) for t in batch)
        b_text, b_mask, b_tag = batch_data
        # Clear the gradient at the start of backward pass
        model.zero_grad()
        # Forward pass and loss calculation
        layer_output = model(b_text, token_type_ids=None, attention_mask=b_mask, labels=b_tag)
        # get the loss
        loss = layer_output[0]
        # backward pass
        loss.backward()
        # add the one epoch loss to total loss
        total_loss += loss.item()
        # clip grad norm, shrink the gradient to avoid exploding gradient
        clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
        # Update the weights and regularization (Parameters update)
        optimizer.step()
        # Update the learning rate
        scheduler.step()

    # Calculate the average loss and add it to the loss list
    average_training_loss = total_loss / len(train_loader)
    print("The average training loss is {}".format(average_training_loss))
    training_loss_values.append(average_training_loss)

    # We have finished the training on one epoch, validation period

    # switch the model to evaluation mode
    model.eval()
    # Set the loss and accuracy to 0
    val_loss, val_accuracy, eval_steps, eval_examples = 0, 0, 0, 0
    predict_tag, original_tag = [], []
    for batch in val_loader:
        batch_data = tuple(t.to(device) for t in batch)
        b_text, b_mask, b_tag = batch_data
        # don't calculate the gradient
        with torch.no_grad():
            # Forward pass and return the predicted tags
            outputs = model(b_text, token_type_ids=None, attention_mask=b_mask, labels=b_tag)

        # Move the predicted and readl tags to CPU
        predict_outcome = outputs[1].detach().cpu().numpy()
        real = b_tag.to('cpu').numpy()

        # Calculate the loss and accuracy
        val_loss += outputs[0].mean().item()
        predict_tag.extend(list(predict) for predict in np.argmax(predict_outcome, axis=2))
        original_tag.extend(real)

    # Calculate the average loss
    average_validation_loss = val_loss / len(val_loader)
    validation_loss_values.append(average_validation_loss)
    print("Average Validation Loss is: {}".format(average_validation_loss))

    pred_tags = [index_to_tag_dict[predicted_index] for predicted, origin in zip(predict_tag, original_tag) for
                 predicted_index, origin_index in zip(predicted, origin) if index_to_tag_dict[origin_index] != "PAD"]
    non_pad_tags = [index_to_tag_dict[origin_index] for origin in original_tag for origin_index in origin if
                    index_to_tag_dict[origin_index] != "PAD"]
    print("Validation Accuracy: {}".format(accuracy_score(pred_tags, non_pad_tags)))
    print("Validation F1-Score: {}\n".format(f1_score(pred_tags, non_pad_tags)))

    # Save our trained model
    if not os.path.exists("./models/"):
        os.mkdir("./models/")
    torch.save(model, "./models/model.pkl")

    # Do some visualization, set the style and the font size, figure size
    sns.set_style(style="darkgrid")
    sns.set(font_scale=1.5)
    plt.rcParams["figure.figsize"] = (30, 15)

    # Plot the learning curve
    plt.plot(training_loss_values, 'b-o', label="Training Loss Curve")
    plt.plot(validation_loss_values, 'r-o', label="Validation Loss Curve")

    # Label the plot
    plt.title("Learning Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
