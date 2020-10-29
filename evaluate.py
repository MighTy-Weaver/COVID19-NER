import json
import pandas as pd
import pickle as pkl
import numpy as np


# Provided function to test accuracy
# You could check the validation accuracy to select the best of your models
def calc_accuracy(preds, tags, padding_id="_t_pad_"):
    """
        Input:
            preds (np.narray): (num_data, length_sentence)
            tags  (np.narray): (num_data, length_sentence)
        Output:
            Proportion of correct prediction. The padding tokens are filtered out.
    """
    preds_flatten = preds.flatten()
    tags_flatten = tags.flatten()
    non_padding_idx = np.where(tags_flatten!=padding_id)[0]
    
    return sum(preds_flatten[non_padding_idx]==tags_flatten[non_padding_idx])/len(non_padding_idx)

def evaluate(pred_file, ground_file):
    file_dict = pkl.load(open(ground_file, "rb"))
    file_preds = pd.read_csv(pred_file)
    return calc_accuracy(np.array([json.loads(line) for line in file_preds["labels"]]), 
              np.array(file_dict["tag_seq"]))
    