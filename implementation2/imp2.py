import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import re

# Importing the dataset
imdb_data = pd.read_csv('IMDB.csv', delimiter=',')


def clean_text(text):

    # remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # remove the characters [\], ['] and ["]
    text = re.sub(r"\\", "", text)
    text = re.sub(r"\'", "", text)
    text = re.sub(r"\"", "", text)
    #pattern = r'[^a-zA-z0-9\s]'
    #text = re.sub(pattern, '', text)

    # convert text to lowercase
    text = text.strip().lower()

    # replace punctuation characters with spaces
    filters = '!"\'#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    translate_dict = dict((c, " ") for c in filters)
    translate_map = str.maketrans(translate_dict)
    text = text.translate(translate_map)

    return text


# this vectorizer will skip stop words
vectorizer = CountVectorizer(
    stop_words="english",
    preprocessor=clean_text,
    max_features=2000
)

# fit the vectorizer on the text
vectorizer.fit(imdb_data['review'])

print(vectorizer)

# get the vocabulary
inv_vocab = {v: k for k, v in vectorizer.vocabulary_.items()}
vocabulary = [inv_vocab[i] for i in range(len(inv_vocab))]
#print(vocabulary)
print(len(vocabulary))

# Get label data and clean it for use

class_data = pd.read_csv('IMDB_labels.csv', delimiter=',')
labels = class_data['sentiment']

def clean_labels(labels):
    ret = np.zeros(len(labels))
    count = 0
    for label in labels:
        if label == 'positive':
            ret[count] = 1
        count += 1 
    
    return ret

labels = clean_labels(labels)
training_data = vectorizer.transform(imdb_data['review'][:30000])
validation_data = vectorizer.transform(imdb_data['review'][30000:40000])

training_labels = labels[:30000]
validation_labels = labels[30000:]
print(training_data.shape[0], len(training_labels))
print(validation_data.shape[0], len(validation_labels))
