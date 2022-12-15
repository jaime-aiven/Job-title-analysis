import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sentence_normalizer import normalize_sentence


# Load tsv file into dataframe
# data = pd.read_csv('data_sets/classified_titles.tsv', sep='\t') # This is the original code
data = pd.read_csv('data_sets/Wseniority_1000.csv') # This is my file
data = pd.read_csv('data_sets/Wseniority.csv') # This is my file


positions_categories = pd.Categorical(data["Classification"])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    data["Position"],
    positions_categories.codes,
    train_size=0.85,
    stratify=positions_categories.codes
)

X_train = [normalize_sentence(x) for x in X_train]
X_test = [normalize_sentence(x) for x in X_test]


# This code is useful to debug bad values in the feature vectors.
#Comment the blocks above that split the dataset and run this to locate bad values.
# The values that will mess this up are ' " NA n/a numbers etc.
# X_train = data["Position"]
# X_test = data["Classification"]
# i=0

# for x in X_train:
#     X_train2 = normalize_sentence(x)
#     i = i+1
#     print(x)
#     print(i)
    
# for x in X_test:
#     X_test2 = normalize_sentence(x)
#     i = i+1
#     print(x)
#     print(i)


with open('data_sets/x_train.pkl', 'wb') as file:
    pickle.dump(X_train, file)

with open('data_sets/x_test.pkl', 'wb') as file:
    pickle.dump(X_test, file)

with open('data_sets/y_train.pkl', 'wb') as file:
    pickle.dump(y_train, file)

with open('data_sets/y_test.pkl', 'wb') as file:
    pickle.dump(y_test, file)

with open('data_sets/positions_categories.pkl', 'wb') as file:
    pickle.dump(positions_categories, file)
