import json

import pandas as pd

answer = pd.read_csv("test_preds.csv", index_col=0)
for i in range(len(answer)):
    if len(json.loads(answer.loc[i, 'labels'])) != 128:
        print("Bad length at {}".format(i))
    else:
        pass
print("end")
