import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn.feature_extraction.text as fe
from nltk.tokenize import TweetTokenizer
import lightgbm as lgb
import nltk
from hyperopt import hp, tpe, Trials, fmin, STATUS_OK
import csv

nltk.download('vader_lexicon')

print('Loading Data')
# Load data
df_train = pd.read_csv('kaggle/input/train.csv')
df_test = pd.read_csv('kaggle/input/test.csv')
df_submission = pd.read_csv('kaggle/input/sample_submission.csv')

df_train['Num_words_text'] = df_train['text'].apply(lambda x:len(str(x).split()))
df_test['Num_words_text'] = df_test['text'].apply(lambda x: len(str(x).split()))
# We don't want to train on the tweets that have less than 3 words in them, since will we just use the full
# text in that case
df_train = df_train[df_train['Num_words_text']>=3]


print('Creating ngrams')
def ngram_features(string):
    if len(string.split()) > 10:
        n = 10
    else:
        n = len(string.split())
    vect = fe.CountVectorizer(ngram_range=(1,n), tokenizer=TweetTokenizer().tokenize)
    vect.fit([string])
    return vect.get_feature_names()


df_train['ngrams'] = df_train['text'].apply(lambda x: ngram_features(str(x)))

# For each tweet calculate the tweet-level features
from nltk.sentiment.vader import SentimentIntensityAnalyzer

print('Tweet level features')
def sentiment_numerical(sentiment):
    if sentiment == 'neutral':
        return 0
    elif sentiment == 'positive':
        return 1
    else:
        return -1
# Tweet length - Done before in the data sep phase, column: Num_words_text

# Sentiment, encoded as {Positive = 1, Neutral = 0, negative = -1}
df_train['num_sent'] = df_train['sentiment'].apply(lambda x: sentiment_numerical(x))

# NLTK Vader sentiments
sid = SentimentIntensityAnalyzer()

def vader(string):
    return [x[1] for x in sid.polarity_scores(string).items()]

df_train['tweet_vader'] = df_train['text'].apply(lambda x: vader(x))

# n-gram lengths
df_train['ngram_lengths'] = df_train.apply(lambda x: [len(j.split()) for j in x['ngrams']], axis=1)
# For each n-gram in each tweet calculate the n-gram level features
print('Ngram level features')
# n-gram lengths / tweet length

df_train['ngram_proportions'] = df_train.apply(lambda x: [j / x['Num_words_text'] for j in x['ngram_lengths']], axis=1)

# NLTK Vader sentiments
df_train['ngram_sentiments'] = df_train.apply(lambda x: [vader(j) for j in x['ngrams']], axis=1)

# For each tweet calculate the non-gram level features for each n-gram
print('Non-ngram level features')
# Tweet length minus n-gram lengths
df_train['non_ngram_lengths'] = df_train.apply(lambda x: [x['Num_words_text'] - n for n in x['ngram_lengths']], axis=1)
# Proportion

df_train['non_ngram_proportions'] = df_train.apply(lambda x: [j / x['Num_words_text'] for j in x['non_ngram_lengths']], axis=1)

# Calculate all Jaccard similarity between each n-gram and the target
# Taking longer than usual? Maybe look into batching it?
def jaccard(string, target):
    a = set(string.lower().split())
    b = set(target.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

print('Jaccard scores')
df_train['jaccard_ngrams'] = df_train.apply(lambda x: [jaccard(y, x['selected_text']) for y in x['ngrams']], axis=1)

print('data wrangling')
temp_ngram_lengths = []
temp_ngram_proportions = []
temp_ngram_sent1 = []
temp_ngram_sent2 = []
temp_ngram_sent3 = []
temp_ngram_sent4 = []
temp_non_ngram_lengths = []
temp_non_ngram_proportions = []
temp_ngram_jaccards = []

temp_tweet_lengths = []
temp_tweet_sent = []
temp_tweet_vader1 = []
temp_tweet_vader2 = []
temp_tweet_vader3 = []
temp_tweet_vader4 = []

for tweet in df_train.itertuples():
    length_t = tweet[5]
    sent_t = tweet[7]
    vader_t = tweet[8]
    
    for length, proportion, sent, non_length, non_proportion, jaccard in zip(tweet[9], tweet[10], tweet[11], tweet[12], tweet[13], tweet[14]):
        # tweet level
        temp_tweet_lengths.append(length_t)
        temp_tweet_sent.append(sent_t)
        temp_tweet_vader1.append(vader_t[0])
        temp_tweet_vader2.append(vader_t[1])
        temp_tweet_vader3.append(vader_t[2])
        temp_tweet_vader4.append(vader_t[3])
        print(sent)
        # ngram level
        temp_ngram_lengths.append(length)
        temp_ngram_proportions.append(proportion)
        temp_ngram_sent1.append(sent[0])
        temp_ngram_sent2.append(sent[1])
        temp_ngram_sent3.append(sent[2])
        temp_ngram_sent4.append(sent[3])
        temp_non_ngram_lengths.append(non_length)
        temp_non_ngram_proportions.append(non_proportion)
        temp_ngram_jaccards.append(jaccard)
# Need to do this for the others as well? Also need to manage the features to fit in the memory
df_train_tabular = pd.DataFrame(list(
                                zip(
                                    temp_tweet_lengths,
                                    temp_tweet_sent,
                                    temp_tweet_vader1,
                                    temp_tweet_vader2,
                                    temp_tweet_vader3,
                                    temp_tweet_vader4,
                                    temp_ngram_lengths,
                                    temp_ngram_proportions,
                                    temp_ngram_sent1,
                                    temp_ngram_sent2,
                                    temp_ngram_sent3,
                                    temp_ngram_sent4,
                                    temp_ngram_jaccards
                                )),
                                columns=[
                                    'tweet_length',
                                    'tweet_sentiment',
                                    'tweet_vader_1',
                                    'tweet_vader_2',
                                    'tweet_vader_3',
                                    'tweet_vader_4',
                                    'ngram_length',
                                    'ngram_proportion',
                                    'ngram_sentiment_1',
                                    'ngram_sentiment_2',
                                    'ngram_sentiment_3',
                                    'ngram_sentiment_4',
                                    'ngram_jaccard'
                                ])


print('Training')

y_train = df_train_tabular['ngram_jaccard'].to_numpy()
X_train = df_train_tabular.drop('ngram_jaccard', axis=1).to_numpy()

train_data = lgb.Dataset(X_train, label=y_train)
# Train the model
final run
param = {'num_leaves': 70, 'objective': 'regression', 'verbose':0, 'boosting': 'gbdt', 'learning_rate': 0.05}
num_rounds = 100000
bst = lgb.train(param, train_data, num_rounds, valid_sets=[validation_data], early_stopping_rounds=5)

"""
Hyperparameter Optimization Code

N_FOLDS = 10

def objective(params, n_folds=N_FOLDS):

  for parameter_name in ['num_leaves', 'subsample_for_bin', 'min_child_samples']:
        params[parameter_name] = int(params[parameter_name])
  cv_results = lgb.cv(params, train_data, nfold=n_folds, num_boost_round=100,
                      early_stopping_rounds=10, metrics='l1', stratified=False)

  best_score = max(cv_results['l1-mean'])

  loss = best_score

  of_connection = open(out_file, 'a')
  writer = csv.writer(of_connection)
  writer.writerow([loss, params])

  return {'loss': loss, 'params': params, 'status': STATUS_OK}

num_leaves = {'num_leaves': hp.quniform('num_leaves', 30, 150, 1)}
learning_rate = {'learning_rate': hp.loguniform('learning_rate', np.log(0.005), np.log(0.2))}

space = {
  'num_leaves': hp.quniform('num_leaves', 30, 150, 1),
  'learning_rate': hp.loguniform('learning_rate', np.log(0.005), np.log(0.2)),
  'subsample_for_bin': hp.quniform('subsample_for_bin', 20000, 300000, 20000),
  'min_child_samples': hp.quniform('min_child_samples', 20, 500, 5),
  'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
  'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
  'colsample_bytree': hp.uniform('colsample_by_tree', 0.6, 1.0)
}

bayes_trials = Trials()

out_file = 'gbm_triasl.csv'
of_connection = open(out_file, 'w')
writer = csv.writer(of_connection)
writer.writerow(['loss', 'params'])
of_connection.close()

MAX_EVALS = 300

best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=MAX_EVALS, trials=bayes_trials)


"""

# Apologies for ugly code, in a rush...
temp_ngram = []
temp_ngram_lengths = []
temp_ngram_proportions = []
temp_ngram_sent1 = []
temp_ngram_sent2 = []
temp_ngram_sent3 = []
temp_ngram_sent4 = []
temp_non_ngram_lengths = []
temp_non_ngram_proportions = []
temp_ngram_jaccards = []

temp_texts = []
temp_tweet_lengths = []
temp_tweet_sent = []
temp_tweet_vader1 = []
temp_tweet_vader2 = []
temp_tweet_vader3 = []
temp_tweet_vader4 = []

for tweet in df_test.itertuples():
    text_t = tweet[2]
    length_t = tweet[4]
    sent_t = tweet[6]
    vader_t = tweet[7]
    
    for ngram, length, proportion, sent, non_length, non_proportion in zip(tweet[5], tweet[8], tweet[9], tweet[10], tweet[11], tweet[12]):
        # tweet level
        temp_texts.append(text_t)
        temp_tweet_lengths.append(length_t)
        temp_tweet_sent.append(sent_t)
        temp_tweet_vader1.append(vader_t[0])
        temp_tweet_vader2.append(vader_t[1])
        temp_tweet_vader3.append(vader_t[2])
        temp_tweet_vader4.append(vader_t[3])
        
        # ngram level
        temp_ngram.append(ngram)
        temp_ngram_lengths.append(length)
        temp_ngram_proportions.append(proportion)
        temp_ngram_sent1.append(sent[0])
        temp_ngram_sent2.append(sent[1])
        temp_ngram_sent3.append(sent[2])
        temp_ngram_sent4.append(sent[3])
        temp_non_ngram_lengths.append(non_length)
        temp_non_ngram_proportions.append(non_proportion)
    

df_test_tabular = pd.DataFrame(list(
                                zip(
                                    temp_texts,
                                    temp_ngram,
                                    temp_tweet_lengths,
                                    temp_tweet_sent,
                                    temp_tweet_vader1,
                                    temp_tweet_vader2,
                                    temp_tweet_vader3,
                                    temp_tweet_vader4,
                                    temp_ngram_lengths,
                                    temp_ngram_proportions,
                                    temp_ngram_sent1,
                                    temp_ngram_sent2,
                                    temp_ngram_sent3,
                                    temp_ngram_sent4
                                )),
                                columns=[
                                    'text',
                                    'ngram',
                                    'tweet_length',
                                    'tweet_sentiment',
                                    'tweet_vader_1',
                                    'tweet_vader_2',
                                    'tweet_vader_3',
                                    'tweet_vader_4',
                                    'ngram_length',
                                    'ngram_proportion',
                                    'ngram_sentiment_1',
                                    'ngram_sentiment_2',
                                    'ngram_sentiment_3',
                                    'ngram_sentiment_4'
                                ])


def predict(tab_data, model):
    ngrams = tab_data['ngram'].to_numpy()
    test_data = tab_data.drop('ngram', axis=1).to_numpy()
    predictions = model.predict(test_data, num_iteration=model.best_iteration)
    
    return ngrams[np.argmax(predictions)]
    
selected_texts = []
for row in df_test.itertuples():
    text = row.text
    output_str = ''
    if row.sentiment == 'neutral' or len(text.split()) <= 2:
        selected_texts.append(text)
    else:
        selected_texts.append(predict(df_test_tabular[df_test_tabular.text == text].drop(['text'], axis=1), bst))


df_submission['selected_text'] = selected_texts
df_submission.to_csv('submission.csv', index=False)
display(df_submission.head(10))