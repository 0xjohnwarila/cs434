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
    # Encodes positive as 1, and negative as 0
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
    positive = vec.transform(document_set[1]).toarray()
    negative = vec.transform(document_set[0]).toarray()

    # Find the total word count for each corpus
    positive_count = np.sum(positive, axis=0)
    negative_count = np.sum(negative, axis=0)

    # Find the numerator for the calculation, adding alpha for laplace
    # smoothing
    positive = positive_count + alpha
    negative = negative_count + alpha

    # Find the denominator, the sum of all words in each corpus plus the number
    # of words in the vocabulary times alpha
    smoothing_denom_pos = np.sum(positive_count) + (len(vocab)*alpha)
    smoothing_denom_neg = np.sum(negative_count) + (len(vocab)*alpha)

    # Divide numerator by denominator, this is now the probability for each
    # word given the class
    positive = np.true_divide(positive, smoothing_denom_pos)
    negative = np.true_divide(negative, smoothing_denom_neg)

    return priors, (negative, positive)


def predict(probs, doc):
    """Predict returns the naive bayes prediction for a document"""
    ret = 1

    # For every word in the document, find the product of the probabilities,
    # raised by the frequency of the word
    for prob, word in zip(probs, doc):
        ret *= prob ** word

    # Return the total probability of the document's words being in the class
    return ret


def validate(probs, priors, data, labels, tag):
    """validate tests the entire set of data and labels, and prints the percent
    correct"""

    # Format document for use
    doc = data[0].toarray()
    count = 0
    predictions = np.zeros(len(labels))
    itr = 0

    for doc, label in zip(data, labels):
        pos = priors[1] * predict(probs[1], doc.toarray().flatten())
        neg = priors[0] * predict(probs[0], doc.toarray().flatten())

        if pos > neg and label == 1:
            count += 1
        elif pos < neg and label == 0:
            count += 1
        if pos > neg:
            predictions[itr] = 1
        itr += 1
    data_out = pd.DataFrame(predictions)
    if tag == "test-default":
        data_out.to_csv("test-prediction1.csv", header=None, index=None)
    elif tag == "test-alpha":
        data_out.to_csv("test-prediction2.csv", header=None, index=None)
    elif tag == "test-best":
        data_out.to_csv("test-prediction3.csv", header=None, index=None)
    else:
        data_out.to_csv("training_validation.csv", header=None, index=None)

    print(tag, 100 * (count / len(labels)))


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
    vocabulary = {v: k for k, v in vectorizer.vocabulary_.items()}
    vocabulary = [vocabulary[i] for i in range(len(vocabulary))]

    training_data = imdb_data['review'][:30000]
    validation_data = imdb_data['review'][30000:40000]

    # Get label data and clean it for use

    labels = clean_labels(pd.read_csv(labels, delimiter=',')['sentiment'])

    training_labels = labels[:30000]
    validation_labels = labels[30000:40000]

    # Run the training algorithm
    priors, probs = training(
        training_data,
        training_labels,
        vocabulary,
        alpha,
        vectorizer
    )

    # Training run
    validate(
        probs,
        priors,
        vectorizer.transform(training_data),
        training_labels,
        'training'
    )

    # Validation run
    validate(
        probs,
        priors,
        vectorizer.transform(validation_data),
        validation_labels,
        'validation'
    )

    


def run_test(data, labels, alpha, max_features, max_df, min_df, tag):
    """run_test runs the algorithm on the testing data"""
    imdb_data = pd.read_csv(data, delimiter=',')

    vectorizer = CountVectorizer(
        stop_words='english',
        preprocessor=clean_text,
        max_features=max_features,
        max_df=max_df,
        min_df=min_df
    )

    # get training/testing data and vocabulary
    vectorizer.fit(imdb_data['review'])
    vocabulary = {v: k for k, v in vectorizer.vocabulary_.items()}
    vocabulary = [vocabulary[i] for i in range(len(vocabulary))]

    training_data = imdb_data['review'][:30000]
    testing_data = imdb_data['review'][40000:]

    # get labels
    labels = clean_labels(pd.read_csv(labels, delimiter=',')['sentiment'])
    training_labels = labels[:30000]
    testing_labels = labels[40000:]

    # Train
    priors, probs = training(
        training_data,
        training_labels,
        vocabulary,
        alpha,
        vectorizer
    )

    # Run testing data
    validate(
        probs,
        priors,
        vectorizer.transform(testing_data),
        testing_labels,
        tag
    )


# Arguments parsing

PARSER = argparse.ArgumentParser(description='Multinomial Naive Bayes')
PARSER.add_argument('data', help='csv file with reviews')
PARSER.add_argument('labels', help='csv file with labels for data')
PARSER.add_argument('run_type', help='See README for run_types')
PARSER.add_argument('--alpha',
                    default=1,
                    type=float,
                    help='alpha value/laplace smoothing')
PARSER.add_argument('--max_features',
                    default=2000,
                    type=int,
                    help='Maximum features for BOW')
PARSER.add_argument('--max_df',
                    default=1,
                    type=float,
                    help='max_df for BOW (float form)')
PARSER.add_argument('--min_df', default=1,
                    type=float,
                    help='min_df for BOW (float form)')

# BEST parameters right now: Alpha = 1, max_features=368, max_df=0.34,
# min_df=0.037
ARGS = PARSER.parse_args()

alphas = np.arange(ARGS.max_df, ARGS.min_df, 0.2)

for alpha in alphas:
    print("----- RUNNING WITH alpha =", alpha, "-----") 
    print("-")
    run(ARGS.data, ARGS.labels, alpha, 2000, 1.0, 1)
    print("-")

exit()

# Switch on the run_type
if ARGS.run_type == "validate_default":
    run(ARGS.data, ARGS.labels, 2, 2000, 1.0, 1)
elif ARGS.run_type == "validate_best":
    run(ARGS.data, ARGS.labels, 1, 368, 0.34, 0.037)
elif ARGS.run_type == "test_default":
    print("Make sure all 50k labels are in the labels file")
    run_test(ARGS.data, ARGS.labels, 1, 2000, 1.0, 1, "test_default")
elif ARGS.run_type == "test_alpha":
    print("Make sure all 50k labels are in the labels file")
    run_test(ARGS.data, ARGS.labels, 1, 2000, 1.0, 1, "test_alpha")
elif ARGS.run_type == "test_best":
    print("Make sure all 50k labels are in the labels file")
    run_test(ARGS.data, ARGS.labels, 1, 2000, 0.34, 0.037, "test_best")
elif ARGS.run_type == "cust":
    print("Custom run, training and validation only")
    run(ARGS.data, ARGS.labels, ARGS.alpha, ARGS.max_features, ARGS.max_df,
        ARGS.min_df)
else:
    print("Unknown runtype, please use on of these:")
    print("validate_default, validate_best, test_default, test_best")


# Below this line is leftover tuning scripts. Left for posterity

"""
max_dfs = np.arange(.3, 0.4, 0.01)
min_dfs = np.arange(0.02, 0.05, 0.001)
alphas = np.arange(0, 5, 0.2)
"""

# alphas = np.arange(0, 2, 0.2)


# for alpha in alphas:
#     print("----- RUNNING WITH alpha =", alpha, "-----") 
#     print("-")
#     run(ARGS.data, ARGS.labels, alpha, ARGS.max_features, ARGS.max_df, 
#         ARGS.min_df)
#     print("-")

"""
for max_df in max_dfs:
    print("----- RUNNING WITH max_df =", max_df, "-----")
    print("-")
    run(ARGS.data, ARGS.labels, ARGS.alpha, ARGS.max_features, max_df,
            ARGS.min_df)
    print("-")
"""

"""
for min_df in min_dfs:
    print("----- RUNNING WITH min_df =", min_df, "-----")
    print("-")
    run(ARGS.data, ARGS.labels, ARGS.alpha, ARGS.max_features, ARGS.max_df,
            min_df)
    print("-")
"""

"""
max_features_list = np.arange(366, 376, 1)

for max_features in max_features_list:
    print("----- RUNNING WITH max_features =", max_features, "-----")
    print("-")
    run(ARGS.data, ARGS.labels, ARGS.alpha, max_features, ARGS.max_df,
            ARGS.min_df)
    print("-")
"""

