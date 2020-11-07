import os

import numpy as np

files = os.listdir("./results/")
print(files)
best_acc = 0
best_acc_name = ""
for file in files:
    if "validation_accuracy" in file:
        accuracy = np.load("./results/{}".format(file))
        for i in range(len(accuracy)):
            if best_acc < accuracy[i]:
                best_acc = accuracy[i]
                best_acc_name = file + "_with_epoch_{}".format(i)
print(best_acc, best_acc_name)
