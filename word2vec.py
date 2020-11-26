# We use word2vec instead of glove embedding in this file
# This word2vec is a self-trained one
import argparse
import json
import os
import pickle
from itertools import chain

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas
import seaborn as sns
from gensim.models import Word2Vec
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Bidirectional, Embedding, Dropout, SpatialDropout1D, Dense, LSTM, \
    BatchNormalization
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.utils.vis_utils import plot_model
from tensorflow.python.ops.init_ops import Constant
from tqdm import trange

# Set up a argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=300, required=False)
parser.add_argument("--bs", type=int, default=64, required=False)
parser.add_argument("--lr", type=float, default=0.001, required=False)
parser.add_argument("--model", type=str, choices=["lstm_bilstm", "bilstm", "bilstm_bilstm"], default="bilstm_bilstm",
                    required=False, help="The model to train the NER")
parser.add_argument("--layers", type=int, default=2, required=False, help="The number of BiLSTM layers you want to try")
args = parser.parse_args()
print(args)

# Set up some parameter we can use
epochs = args.epoch
BS = args.bs
LR = args.lr

# Load the data for three splits
train_dict = pickle.load(open("./data/train.pkl", 'rb'))
val_dict = pickle.load(open("./data/val.pkl", 'rb'))
test_dict = pickle.load(open("./data/test.pkl", 'rb'))

total_list = train_dict["word_seq"] + val_dict["word_seq"] + test_dict["word_seq"]
model = Word2Vec(total_list, size=300, window=5, min_count=1, workers=4)
# model.train(val_dict["word_seq"], total_examples=2950, epochs=5)
# model.train(test_dict["word_seq"], total_examples=2950, epochs=5)
print("Train word2vec model down.")

# Give all the words appeared in our corpus their glove embedding, for those who are not exist, random initialize them
encoded_dict = {}
count = 0
total = 0
word2vec_keys = model.wv.vocab.keys()
dimension = 300
for i in [train_dict, val_dict, test_dict]:
    for j in trange(len(i['word_seq'])):
        for word in i['word_seq'][j]:
            if word not in word2vec_keys:
                encoded_dict[word] = np.random.rand(1, dimension)[0]
                count += 1
                total += 1
            else:
                encoded_dict[word] = model.wv[word]
                total += 1
# Test how many words are found in glove and how many are randomly initialized
print("words not found {}".format(count))
print("words total {}".format(total))
print(len(encoded_dict))
if not os.path.exists('./word2vec'):
    os.mkdir('./word2vec/')
np.save("./word2vec/encoded_dict_{}d.npy".format(dimension), encoded_dict)

# Build a dict that records the word to a single unique integer, and our encoded matrix for word embedding
encoded_word2id = {}
encoded_matrix = np.zeros((len(encoded_dict.keys()), dimension), dtype=float)
for i, word in enumerate(encoded_dict.keys()):
    encoded_word2id[word] = i
    encoded_matrix[i] = encoded_dict[word]
print(encoded_matrix.shape)
np.save("./word2vec/encoded_matrix_{}d.npy".format(dimension), encoded_matrix)

# Build the tag <--> index dictionary and add PAD tag into it
tag_list = list(set(chain(*train_dict["tag_seq"])))
tag_to_index_dict = {t: i for i, t in enumerate(tag_list)}
index_to_tag_dict = {i: t for i, t in enumerate(tag_list)}

# save out dictionary for generation
if not os.path.exists('./lstm_model'):
    os.mkdir('./lstm_model/')
if not os.path.exists('./lstm_results'):
    os.mkdir('./lstm_results')
np.save("./lstm_model/model_tag2id_e{}_bs{}.npy".format(epochs, BS), tag_to_index_dict)
np.save("./lstm_model/model_id2tag_e{}_bs{}.npy".format(epochs, BS), index_to_tag_dict)

# Load some parameters for deep learning
embedding_dim = dimension
num_words = len(encoded_dict)
input_length = 128
n_tags = len(tag_to_index_dict)
print(embedding_dim, num_words, input_length, n_tags)


# Set our model
def get_bi_lstm_model():
    model = Sequential()
    model.add(
        Embedding(num_words, embedding_dim, embeddings_initializer=Constant(encoded_matrix), input_length=input_length,
                  trainable=True))
    model.add(SpatialDropout1D(0.2))
    model.add(BatchNormalization())
    if args.model == "lstm_bilstm":
        model.add(LSTM(128, return_sequences=True))
    elif args.model == "bilstm_bilstm":
        for _ in range(args.layers):
            model.add(Bidirectional(LSTM(128, return_sequences=True)))
            model.add(BatchNormalization())
            model.add(Dropout(0.15))
    adam = Adam(lr=LR, beta_1=0.9, beta_2=0.999)
    model.add(Dense(units=n_tags, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    model.summary()
    return model


# Define the function to train our model
def train_model(X, y, val_X, val_y, model):
    hist = model.fit(X, y, batch_size=BS, verbose=1, epochs=epochs, validation_data=(val_X, val_y), shuffle=True)
    return hist


# build our model and print the summary
model_bi_lstm_lstm = get_bi_lstm_model()
try:
    plot_model(model_bi_lstm_lstm, show_shapes=True)
except ImportError:
    pass

# Use the dict we've prepared before to do the embedding and transformation
train_input = np.array(
    [[encoded_word2id[word] for word in train_dict['word_seq'][i]] for i in range(len(train_dict['word_seq']))])
val_input = np.array(
    [[encoded_word2id[word] for word in val_dict['word_seq'][i]] for i in range(len(val_dict['word_seq']))])
test_input = np.array(
    [[encoded_word2id[word] for word in test_dict['word_seq'][i]] for i in range(len(test_dict['word_seq']))])
train_output = np.array(
    [[tag_to_index_dict[tag] for tag in train_dict['tag_seq'][i]] for i in range(len(train_dict['tag_seq']))])
val_output = np.array(
    [[tag_to_index_dict[tag] for tag in val_dict['tag_seq'][i]] for i in range(len(val_dict['tag_seq']))])

# Check the shape of our input, their first dimension must be the same
print(train_input.shape, val_input.shape, test_input.shape)
print(train_output.shape, val_output.shape)

# Train our model and save the loss recording
history = train_model(train_input, train_output, val_input, val_output, model_bi_lstm_lstm)

# Do some visualization
sns.set_style(style="darkgrid")
sns.set(font_scale=1.75)
plt.rcParams["figure.figsize"] = (30, 15)

mpl.use('Agg')
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
# plt.show()
plt.savefig(
    './lstm_results/accuracy_BS{}E{}LR{}_{}d_{}layer.png'.format(BS, epochs, LR, dimension,
                                                                 args.layers))
plt.clf()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
# plt.show()
plt.savefig(
    './lstm_results/model_loss_BS{}E{}LR{}_{}d_{}layers.png'.format(BS, epochs, LR,
                                                                    dimension, args.layers))
print("save images down.")
# Save the validation accuracy for us to find the best model trained
np.save(
    './lstm_results/model_results_val_BS{}E{}LR{}_{}d_{}layers.npy'.format(BS, epochs, LR,
                                                                           dimension, args.layers),
    history.history['val_accuracy'])
print("save history validation data down.")
# Save our trained model and open up a answer csv, initialize all the id
# try:
#     model_bi_lstm_lstm.save(
#     './lstm_model/model_BS{}E{}LR{}_{}d_{}layers.pkl'.format(BS, epochs, LR, dimension, args.layers))
# except:
#     pass

answer = pandas.DataFrame(columns=['id', 'labels'])
answer['id'] = test_dict['id']

# Predict on the test dict and save it to answer csv file
predict = model_bi_lstm_lstm.predict(test_input)
for i in range(len(answer)):
    sentence_tag = []
    for j in range(128):
        tag = index_to_tag_dict[np.argmax(predict[i][j])]
        sentence_tag.append(tag)
    answer.loc[i, 'labels'] = json.dumps(sentence_tag)
answer.to_csv(
    './lstm_results/answer_BS{}E{}LR{}_{}d_{}layers.csv'.format(BS, epochs, LR, dimension,
                                                                args.layers), index=True)
print("save results csv down.")
