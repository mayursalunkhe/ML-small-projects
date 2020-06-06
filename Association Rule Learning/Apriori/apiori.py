# apiori
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Importing dataset
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
transaction = []
for i in range(0, 7501):
    transaction.append([str(dataset.values[i, j]) for j in range(0, 20)])


# Training Apriori on dataset
from apyori import apriori
rules = apriori(transaction,min_support = 0.0028, min_confidence = 0.2,
                min_lift = 3, min_length = 2)

'''
min_support - look for items atleast buy 3 or 4 times a day,
              say item purchased 3 times a day so for 1 week its 3*7
              out of 7500 items so calculation is (3*7/7500)

min_confidence - constant in 0.2 (20%), ie our rule will be right atleast 20% of the time.
min_length - rules composed atleast 2 products
'''
results = list(rules)

results_list = []
for i in range(0, len(results)):
    results_list.append('RULE:\t' + str(results[i][0]) + '\nSUPPORT:\t' + str(results[i][1]) + '\nInfo:\t' + str (results[i][2]))

