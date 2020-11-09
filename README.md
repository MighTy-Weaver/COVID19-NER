# 2020-Fall COMP4901K-Project COVID-19 Named Entity Recognition 
This is the code for the Group Project of course **Machine Learning for Natural Language Processing (COMP4901K) Fall-2020**

I implemented two approaches to solve the problem.

    Dataset are provided in the data folder for reproduction.

## 1. BERT Classfication 
By fine-tuning the pretrained BERT model from `HuggingFace`, I embedded all the words and
use the provided BERT classification model to do the named entity recognition on the test set.

The models are saved at the `./models/` and the results are saved at `./results/`

**The best results achieved is 89.3% on the validation set.** This is probably because there are some domain knowledge and BERT can't treat them well.

Codes related are provided below, to reproduce, please follow this sequence:

`BERT_finetuning.py` : code for fine-tuning the BERT model with the old version of `BertTokenizer`, which `tokenize` and `convert_token_to_id` is used, followed by classification.

`BERT_revised.py` : code for fine-tuning the BERT model with `transformer 3.4.0`, where`BertTokenizer.encode` is used, followed by a classification.

`evaluate.py` : Sample code provided for evaluating the output and results.

`find_best_model.py` : code for finding the best model according to the validation accuracy.

`loss_visualization.py` : visualizing the training/validation loss/accuracy for all the models.

`predict_verify.py`: code for generating result on the test data and verify the result by checking the generation's length.

`test.py` : useless code for testing and trail purpose.


## 2. Glove + Bi-LSTM 

Since BERT is not performing that excellent in this task, an alternative is the pretrained-word-embedding + LSTM
I used the Glove embedding with 6 Billion tokens in 100 dimensions. Details can be found at: https://nlp.stanford.edu/projects/glove/
Then I used the Bidirectional-LSTM model to do the classification as its capable of capturing some long-term information with the ability of fitting the neural network with nonlinearity.

**The best result achieved is 93.2% on the validation set.** This is high enough for this course project.

The models are saved at `./lstm_models/` and the results, answers as well as the loss recording are all saved at `./lstm_results/`

Codes related are:

`LSTM_NER.py` : The full code to do the embedding, train the model and do the classification. Please check the arguments for detail.

`BiLSTM.ipynb` : A jupyter notebook of coding during the developing process. All codes are compiled together in the `LSTM_NER.py` with code-readability enhanced.

