import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import argparse
from utils import load_data, f1, accuracy_score, load_dictionary, dictionary_info
from tree import DecisionTreeClassifier, RandomForestClassifier
sns.set()

def load_args():

    parser = argparse.ArgumentParser(description='arguments')
    parser.add_argument('--county_dict', default=1, type=int)
    parser.add_argument('--decision_tree', default=1, type=int)
    parser.add_argument('--random_forest', default=1, type=int)
    parser.add_argument('--ada_boost', default=1, type=int)
    parser.add_argument('--root_dir', default='../data/', type=str)
    _args = parser.parse_args()

    return _args


def county_info(_args):
    county_dict = load_dictionary(_args.root_dir)
    dictionary_info(county_dict)

def decision_tree_testing(x_train, y_train, x_test, y_test):
    print('Decision Tree\n\n')
    clf = DecisionTreeClassifier(max_depth=2)
    clf.fit(x_train, y_train)
    preds_train = clf.predict(x_train)
    preds_test = clf.predict(x_test)
    train_accuracy = accuracy_score(preds_train, y_train)
    test_accuracy = accuracy_score(preds_test, y_test)
    print('Train {}'.format(train_accuracy))
    print('Test {}'.format(test_accuracy))
    preds = clf.predict(x_test)
    print('F1 Test {}'.format(f1(y_test, preds)))

def varying_depth_tree_testing(x_train, y_train, x_test, y_test, start, end):
    print("Varying depths, decision tree")

    training_acc = np.zeros(end+1 - start)
    testing_acc = np.zeros(end+1 - start)
    f1_acc = np.zeros(end+1 - start)
    best_test = 0
    best_test_d = 0
    best_f1 = 0
    best_f1_d = 0
    best_gain = 0
    for depth in range(start, end+1):
        clf = DecisionTreeClassifier(max_depth=depth)
        clf.fit(x_train, y_train)
        preds_train = clf.predict(x_train)
        preds_test = clf.predict(x_test)
        train_accuracy = accuracy_score(preds_train, y_train)
        training_acc[depth-1] = train_accuracy
        test_accuracy = accuracy_score(preds_test, y_test)
        if test_accuracy > best_test:
            best_test = test_accuracy
            best_test_d = depth
        testing_acc[depth-1] = test_accuracy
        f1_accuracy = f1(y_test, preds_test)
        if f1_accuracy > best_f1:
            best_f1 = f1_accuracy
            best_f1_d = depth
        f1_acc[depth-1] = f1_accuracy

    print("Best test accuracy is", best_test, "at depth", best_test_d)
    print("Best F1 accuracy is", best_f1, "at depth", best_f1_d)
    plt.axvline(x=best_f1_d, linestyle='dashed')

    # Plotting
    df = pd.DataFrame({
        'x': range(start, end+1),
        'train': training_acc,
        'test': testing_acc,
        'f1': f1_acc
        })

    plt.style.use('seaborn-darkgrid')


    num = 0

    for column in df.drop('x', axis=1):
        num += 1
        plt.plot(df['x'], df[column], marker='',
                 linewidth=1, alpha=0.9, label=column)
    plt.legend(loc=2, ncol=2)
    plt.title("Accuracy at Varying Decision Tree Depths")
    plt.xlabel("Depth")
    plt.ylabel("Accuracy")
    plt.show()

def random_forest_testing(x_train, y_train, x_test, y_test):
    print('Random Forest\n\n')
    rclf = RandomForestClassifier(max_depth=7, max_features=11, n_trees=50)
    rclf.fit(x_train, y_train)
    preds_train = rclf.predict(x_train)
    preds_test = rclf.predict(x_test)
    train_accuracy = accuracy_score(preds_train, y_train)
    test_accuracy = accuracy_score(preds_test, y_test)
    print('Train {}'.format(train_accuracy))
    print('Test {}'.format(test_accuracy))
    preds = rclf.predict(x_test)
    print('F1 Test {}'.format(f1(y_test, preds)))


###################################################
# Modify for running your experiments accordingly #
###################################################
if __name__ == '__main__':
    args = load_args()
    x_train, y_train, x_test, y_test = load_data(args.root_dir)
    if args.county_dict == 1:
        county_info(args)
    if args.decision_tree == 1:
        # decision_tree_testing(x_train, y_train, x_test, y_test)
        varying_depth_tree_testing(x_train, y_train, x_test, y_test, 1, 25)
    if args.random_forest == 1:
        random_forest_testing(x_train, y_train, x_test, y_test)

    print('Done')




