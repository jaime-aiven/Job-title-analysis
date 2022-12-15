# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 10:03:23 2021

@author: JLO033
"""

import pickle
from test_functions import load_instances, test_classifier
import numpy as np


import pandas as pd
from sklearn.model_selection import train_test_split
#from sentence_normalizer import normalize_sentence




# ## PHASE 0: DEFINITIONS

# def normalize_sentence(sentence):
#     # remove ' and .
#     sentence = sentence.replace("'", "")
#     sentence = sentence.replace(".", "")
#     # change , for space
#     sentence = sentence.replace(", ", " ")
#     sentence = sentence.replace(",", " ")
#     sentence = sentence.replace("/", " ")
#     sentence = sentence.lower()
#     return sentence




# #PHASE 1: IMPORT AND PREPARE TRAINING DATA
# # Load tsv file into dataframe
# # This is a file with 10k classified examples
# data = pd.read_csv('C:/Users/JLO033/MIT/6.86x Data Science in Python/Job title analysis/ai-job-title-level-classification-master/data_process/data_sets/Wseniority_1000.csv') 


# positions_categories = pd.Categorical(data["Classification"])

# # Split dataset
# X_train, X_test, y_train, y_test = train_test_split(
#     data["Position"],
#     positions_categories.codes,
#     train_size=0.85,
#     stratify=positions_categories.codes
# )

# X_train = [normalize_sentence(x) for x in X_train]
# X_test = [normalize_sentence(x) for x in X_test]

# #Dump the vectors in files for later recall

# with open('C:/Users/JLO033/MIT/6.86x Data Science in Python/Job title analysis/ai-job-title-level-classification-master/data_process/data_sets/x_train.pkll', 'wb') as file:
#     pickle.dump(X_train, file)

# with open('C:/Users/JLO033/MIT/6.86x Data Science in Python/Job title analysis/ai-job-title-level-classification-master/data_process/data_sets/x_test.pkl', 'wb') as file:
#     pickle.dump(X_test, file)

# with open('C:/Users/JLO033/MIT/6.86x Data Science in Python/Job title analysis/ai-job-title-level-classification-master/data_process/data_sets/y_train.pkl', 'wb') as file:
#     pickle.dump(y_train, file)

# with open('C:/Users/JLO033/MIT/6.86x Data Science in Python/Job title analysis/ai-job-title-level-classification-master/data_process/data_sets/y_test.pkl', 'wb') as file:
#     pickle.dump(y_test, file)

# with open('C:/Users/JLO033/MIT/6.86x Data Science in Python/Job title analysis/ai-job-title-level-classification-master/data_process/data_sets/positions_categories.pkl', 'wb') as file:
#     pickle.dump(positions_categories, file)




# # PHASE 2: TRAIN MODEL



# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_extraction.text import TfidfTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.linear_model import SGDClassifier
# from sklearn.model_selection import GridSearchCV

# from fit_tune_function import fit_tune_store_sgdcv


# clf_pipeline = Pipeline([
#     ('vect', CountVectorizer()),
#     ('tfidf', TfidfTransformer()),
#     ('clf', SGDClassifier(loss='hinge', penalty='l2',  alpha=1e-3, random_state=42, max_iter=5, tol=None)),
# ])

# parameters = {
#     'vect__ngram_range': [(1, 1), (1, 2), (2, 3), (3, 4), ],
#     'tfidf__use_idf': (True, False),
#     'clf__random_state': (1, 21, 33, 42, 88, 100, 160),
#     'clf__alpha': (1e-2, 1e-3, 1e-4, 0.1, 1e-6, ),
#     'clf__max_iter': (2, 5, 10, 20, 100, 200)
# }

# sgd_clf_gscv = GridSearchCV(clf_pipeline, parameters, cv=5, iid=False, n_jobs=-1)
# fit_tune_store_sgdcv(sgd_clf_gscv, 'sgd')





# PHASE 3: PERFORM PREDICTION





#Importing the real data to be predicted


# Load tsv file into dataframe

real_data = pd.read_csv('C:/Users/JLO033/MIT/6.86x Data Science in Python/Job title analysis/Seniority/data_process/data_sets/predict_seniority.csv') # This is my file for prediction
positions_categories2 = pd.Categorical(real_data["Classification"])



#I don't need X_train as I already have a trained classifier from before
X_real = real_data["Position"]

X_real = [normalize_sentence(x) for x in X_real]


## Code for debugging incoming vectors
# i=0

# for x in X_real:
#     X_train2 = normalize_sentence(x)
#     i = i+1
#     print(x)
#     print(i)



with open('data_sets/x_real.pkl', 'wb') as file:
    pickle.dump(X_real, file)
    
    

with open('data_sets/positions_categories.pkl', 'wb') as file:
    pickle.dump(positions_categories2, file)




# Perform prediction


# Load instances
clf, count_vect, tfidf_transformer = load_instances('sgd')


# Test and show results
X_real_counts = count_vect.transform(X_real)
X_real_tfidf = tfidf_transformer.transform(X_real_counts)
real_predict = clf.predict(X_real_tfidf)

np.savetxt('C:/Users/JLO033/MIT/6.86x Data Science in Python/Job title analysis/Seniority/data_process/data_sets/pred_out.txt',real_predict)


test_classifier(clf, real_predict)