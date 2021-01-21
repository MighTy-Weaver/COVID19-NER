# 2020-Fall COMP4901K-Project COVID-19 Named Entity Recognition 
This is the code for the Group Project of course **Machine Learning for Natural Language Processing (COMP4901K) Fall-2020**

Basically speaking, BERT and BiLSTM are the two approaches to solve the task. The final result shows that Double Layer BiLSTM is 
our best choice.

> Dataset are provided in the data folder for reproduction

Use `pip install -r requirements.txt` to install the packages needed. We've tested this repository under Python 3.6 and
CentOS 7.0

## 0. Some non-functional codes:

`evaluate.py` : Sample code provided for evaluating the output and results.


## 1. BERT Classification 
By fine-tuning the pretrained BERT model from `HuggingFace`, I embedded all the words and
use the provided BERT classification model to do the named entity recognition on the test set. Details can be found at: https://huggingface.co/transformers/model_doc/bert.html

The models are saved at the `./models/` and the results are saved at `./results/`

**The best results achieved is 89.3% on the validation set.** This is probably because there are some domain knowledge and BERT can't treat them well.

Codes related are provided below, to reproduce, please follow this sequence:


`BERT_finetuning.py` **(This code is deprecated by `BERT_revised.py`)**: code for fine-tuning the BERT model with the old version of `BertTokenizer`, which `tokenize` and `convert_token_to_id` is used, followed by classification.

`BERT_revised.py` : code for fine-tuning the BERT model with `transformer 3.4.0`, where`BertTokenizer.encode` is used, followed by a classification.

`BERT_find_best_model.py` : code for finding the best model according to the validation accuracy.

`BERT_loss_visualization.py` : visualizing the training/validation loss/accuracy for all the models.

`BERT_predict_verify.py`: code for generating result on the test data and verify the result by checking the generation's length.

## 2. Glove + Bi-LSTM 

Since BERT is not performing that excellent in this task, an alternative is the pretrained-word-embedding + LSTM
I used the Glove embedding with 6 Billion tokens in 100 dimensions. Details can be found at: https://nlp.stanford.edu/projects/glove/
Then I used the Bidirectional-LSTM model to do the classification as its capable of capturing some long-term information with the ability of fitting the neural network with non-linearity.

**The best result achieved is 92.1% on the validation set.** This is high enough for this course project.

The models are saved at `./lstm_models/` and the results, answers as well as the loss recording are all saved at `./lstm_results/`

Codes related are:

`LSTM_NER.py` : The full code to do the embedding, train the model and do the classification. Please check the arguments for detail.

`BiLSTM.ipynb` : A jupyter notebook of coding during the developing process. All codes are compiled together in the `LSTM_NER.py` with code-readability enhanced.

`LSTM_find_best_model.py` : The code to find the best lstm model trained from all the models, it will read the generated accuracy list and find the one with the highest accuracy.

### 3. Glove Update (2020/11/20) :
---
We've added the Glove downloading part into the LSTM_NER.py code and now the code can automatically download all the glove pretrained model.

User can pass a parameter of how many billion words is the glove pretrained on to decide which glove model to use.


### 4. LSTM Structure update (2020/11/24) :
---
We've experimented on two more model structures: BiLSTM + BiLSTM and LSTM + BiLSTM. User can pass a parameter called `model` to decide which 
lstm structure to use.  

We experimented on double layers of Bidirectional LSTM and triple layers of BiLSTM, both networks converged at around 40 epochs, **the double layers BiLSTM achieved the highest accuracy of 93.25%**, which is our best record.

### 5. Word2vec model introduced (2020/11/24) :
---
We introduced a new word2vec model for the embedding rather than Glove. We tried for two times, the first time we use the `word2vec-google-news-300` pretrained model for embedding, it is relatively old and haven't been updated for years but still shows a good results with final accuracy `0.9201`. 

And we are also thinking about that maybe using our own database to train the model is better, since the sentences are all in the specific domain. So we next use the word2vec to train on the given dataset and use this as embedding. The final accuracy is `0.9253`, which is slightly better. 
