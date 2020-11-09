import os

import numpy as np

files = [file for file in os.listdir('./lstm_results/') if "results_val_" in file]
best_acc = [0]
best_acc_name = ""
for acc in files:
    accuracy = np.load('./lstm_results/{}'.format(acc))
    max_accuracy = max(accuracy)
    if max(best_acc) < max_accuracy:
        best_acc = accuracy
        best_acc_name = acc
    else:
        pass
print(
    "The last epoch has the val accuracy of \t{}\nthe highest acc among training is \t{}\nthe name of the model is \t{}".format(
        best_acc[-1], max(best_acc), best_acc_name))
