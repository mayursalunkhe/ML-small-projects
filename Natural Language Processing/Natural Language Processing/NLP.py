# Natural language processing


import numpy as np
import matplotlib as plt
import pandas as pd

dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)


# Cleaning texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i]) # Keep letters and replace space with removed char.
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))] # use set to make it faster
    review = ' '.join(review) # list to string saperate with space
    corpus.append(review)

# Creating bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

from ClassificationFunction import *

naiveBayesClassification(X, y)
LogisticClassification(X, y)
KNN_Classification(X, y)
SvmClassification(X, y)
KernelSvmClassification(X, y)
DecisionTreeClassification(X, y)
RandomForestClassification(X, y)