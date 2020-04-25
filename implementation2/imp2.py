""" Multinomial Naive Bayes """
import re
import argparse
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


def clean_text(text):
    """Clean text of html and set to lowercase"""
    # remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # remove the characters [\], ['] and ["]
    text = re.sub(r"\\", "", text)
    text = re.sub(r"\'", "", text)
    text = re.sub(r"\"", "", text)
    # convert text to lowercase
    text = text.strip().lower()

    # replace punctuation characters with spaces
    filters = '!"\'#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    translate_dict = dict((c, " ") for c in filters)
    translate_map = str.maketrans(translate_dict)
    text = text.translate(translate_map)

    return text


def clean_labels(labels):
    """clean_labels return a list with positive label as 1 and negative as 0"""
    ret = np.zeros(len(labels))
    count = 0
    for label in labels:
        if label == 'positive':
            ret[count] = 1
        count += 1
    return ret


def training(data, labels, vocab, alpha, vec):
    """ Training """

    # Calculate prior probability for the two classes, positive and negative
    priors = np.zeros(2)

    for label in labels:
        priors[int(label)] += 1

    priors[0] /= len(labels)
    priors[1] /= len(labels)

    # Separate the classes
    document_set = [[], []]

    for i in range(data.shape[0]):
        if labels[i] == 0:
            document_set[0].append(data[i])
        else:
            document_set[1].append(data[i])

    # Laplace smoothing
    positive = vec.transform(document_set[1])
    negative = vec.transform(document_set[0])

    positive = positive.toarray()
    negative = negative.toarray()
    positive_1 = np.sum(positive, axis=0)
    negative_1 = np.sum(negative, axis=0)

    positive = positive_1 + alpha
    negative = negative_1 + alpha

    ls_pos = np.sum(positive_1) + (len(vocab)*alpha)
    ls_neg = np.sum(negative_1) + (len(vocab)*alpha)

    positive = np.true_divide(positive, ls_pos)
    negative = np.true_divide(negative, ls_neg)

    return priors, positive, negative


def predict(probs, doc):
    """Predict returns the naive bayes prediction for a document"""
    ret = 1
    doc = doc.flatten()

    for prob, word in zip(probs, doc):
        ret *= prob ** word

    return ret


def validate(positive, negative, priors, data, labels, run):
    """validate tests the entire set of data and labels, and prints the percent
    correct"""
    doc = data[0].toarray()
    count = 0

    for doc, label in zip(data, labels):
        pos = priors[1] * predict(positive, doc.toarray())
        neg = priors[0] * predict(negative, doc.toarray())

        if pos > neg and label == 1:
            count += 1
        elif pos < neg and label == 0:
            count += 1
    print(run, 100 * (count / len(labels)))


def run(data, labels, alpha, max_features, max_df, min_df):
    """run trains the model and validates it off the validation data"""
    # Importing the dataset
    imdb_data = pd.read_csv(data, delimiter=',')

    # this vectorizer will skip stop words
    vectorizer = CountVectorizer(
        stop_words="english",
        preprocessor=clean_text,
        max_features=max_features,
        max_df=max_df,
        min_df=min_df
    )

    # fit the vectorizer on the text
    vectorizer.fit(imdb_data['review'])

    # get the vocabulary
    inv_vocab = {v: k for k, v in vectorizer.vocabulary_.items()}
    vocabulary = [inv_vocab[i] for i in range(len(inv_vocab))]

    # Get label data and clean it for use

    class_data = pd.read_csv(labels, delimiter=',')
    labels = class_data['sentiment']

    labels = clean_labels(labels)
    training_data = imdb_data['review'][:30000]
    validation_data = imdb_data['review'][30000:40000]

    training_labels = labels[:30000]
    validation_labels = labels[30000:]

    priors, positive, negative = training(
        training_data,
        training_labels,
        vocabulary,
        alpha,
        vectorizer
    )
    validate(
        positive, negative, priors,
        vectorizer.transform(validation_data),
        validation_labels,
        'validation'
    )
    validate(
        positive, negative, priors,
        vectorizer.transform(training_data),
        training_labels,
        'training'
    )

PARSER = argparse.ArgumentParser(description='Multinomial Naive Bayes')
PARSER.add_argument('data', help='csv file with reviews')
PARSER.add_argument('labels', help='csv file with labels for data')
PARSER.add_argument('alpha', type=float, help='alpha value for laplace smoothing')
PARSER.add_argument('max_features', type=int, help='Maximum features for BOW')
PARSER.add_argument('max_df', type=float, help='max_df for BOW (float form)')
PARSER.add_argument('min_df', type=float, help='min_df for BOW (float form)')

# BEST parameters right now: Alpha = 1, max_features=368, max_df=0.34,
# min_df=0.037
args = PARSER.parse_args()
run(args.data, args.labels, args.alpha, args.max_features, args.max_df,
        args.min_df)

max_dfs = np.arange(.3, 0.4, 0.01)
min_dfs = np.arange(0.02, 0.05, 0.001)
alphas = np.arange(0, 5, 0.2)

"""
for alpha in alphas:
    print("----- RUNNING WITH alpha =", alpha, "-----") 
    print("-")
    run(args.data, args.labels, alpha, args.max_features, args.max_df, 
        args.min_df)
    print("-")
"""

"""
for max_df in max_dfs:
    print("----- RUNNING WITH max_df =", max_df, "-----")
    print("-")
    run(args.data, args.labels, args.alpha, args.max_features, max_df,
            args.min_df)
    print("-")
"""

"""
for min_df in min_dfs:
    print("----- RUNNING WITH min_df =", min_df, "-----")
    print("-")
    run(args.data, args.labels, args.alpha, args.max_features, args.max_df,
            min_df)
    print("-")
"""

"""
max_features_list = np.arange(366, 376, 1)

for max_features in max_features_list:
    print("----- RUNNING WITH max_features =", max_features, "-----")
    print("-")
    run(args.data, args.labels, args.alpha, max_features, args.max_df,
            args.min_df)
    print("-")
"""

