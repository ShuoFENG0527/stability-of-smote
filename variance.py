import os
import numpy as np
import pandas as pd
import csv
count = 0
variance_dataset = pd.DataFrame()
# single = open('D:\\investigation on stability of smote\\oversamplingtechnique\\variance.csv', 'w', newline='')
# single_writer = csv.writer(single)

for inputfile in os.listdir("D:\\investigation on the stability of smote\\svm"):
    dataset = pd.read_csv("D:\\investigation on the stability of smote\\svm\\" + inputfile)
    dataset = dataset.drop(columns="inputfile")
    variance_column = []
    for i, r in dataset.iterrows():
        r = r[0: 100]
        variance = r.var()
        variance_column.append(variance)

    # variance_column = pd.DataFrame(variance_column)
    variance_dataset[inputfile] = variance_column
print(variance_dataset)
variance_dataset.to_csv(r"D:\\investigation on the stability of smote\\svm variance\\stable_auc_variance.csv")

