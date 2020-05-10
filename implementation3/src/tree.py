import numpy as np

class Node():
    """
    Node of decision tree

    Parameters:
    -----------
    prediction: int
        Class prediction at this node
    feature: int
        Index of feature used for splitting on
    split: int
        Categorical value for the threshold to split on for the feature
    left_tree: Node
        Left subtree
    right_tree: Node
        Right subtree
    """
    def __init__(self, prediction, feature, split, left_tree, right_tree):
        self.prediction = prediction
        self.feature = feature
        self.split = split
        self.left_tree = left_tree
        self.right_tree = right_tree


class DecisionTreeClassifier():
    """
    Decision Tree Classifier. Class for building the decision tree and making
    predictions

    Parameters:
    ------------
    max_depth: int
        The maximum depth to build the tree. Root is at depth 0, a single split
        makes depth 1 (decision stump)
    """

    def __init__(self, max_depth=None, max_features=-1):
        self.max_depth = max_depth
        self.max_features = max_features

    # take in features X and labels y
    # build a tree
    def fit(self, X, y):
        self.num_classes = len(set(y))
        self.root = self.build_tree(X, y, depth=1)

    # make prediction for each example of features X
    def predict(self, X):
        preds = [self._predict(example) for example in X]

        return preds

    # prediction for a given example
    # traverse tree by following splits at nodes
    def _predict(self, example):
        node = self.root
        while node.left_tree:
            if example[node.feature] < node.split:
                node = node.left_tree
            else:
                node = node.right_tree
        return node.prediction

    # accuracy
    def accuracy_score(self, X, y):
        preds = self.predict(X)
        accuracy = (preds == y).sum()/len(y)
        return accuracy

    # function to build a decision tree
    def build_tree(self, X, y, depth):
        num_samples, num_features = X.shape
        # which features we are considering for splitting on
        if self.max_features > 0:
            self.features_idx = np.random.choice(X.shape[1], self.max_features,
                                                 replace=False)
        else:
            self.features_idx = np.arange(0, X.shape[1])

        # store data and information about best split
        # used when building subtrees recursively
        best_feature = None
        best_split = None
        best_gain = 0.0
        best_left_X = None
        best_left_y = None
        best_right_X = None
        best_right_y = None

        # what we would predict at this node if we had to
        # majority class
        num_samples_per_class = [np.sum(y == i) for i in range(self.num_classes)]
        prediction = np.argmax(num_samples_per_class)

        # if we haven't hit the maximum depth, keep building
        if depth <= self.max_depth:
            # consider each feature
            for feature in self.features_idx:
                # consider the set of all values for that feature to split on
                possible_splits = np.unique(X[:, feature])
                for split in possible_splits:
                    # get the gain and the data on each side of the split
                    # >= split goes on right, < goes on left
                    gain, left_X, right_X, left_y, right_y = self.check_split(X, y, feature, split)
                    # if we have a better gain, use this split and keep track of data
                    if gain > best_gain:
                        best_gain = gain
                        best_feature = feature
                        best_split = split
                        best_left_X = left_X
                        best_right_X = right_X
                        best_left_y = left_y
                        best_right_y = right_y
        
        # if we haven't hit a leaf node
        # add subtrees recursively
        if best_gain > 0.0:
            left_tree = self.build_tree(best_left_X, best_left_y, depth=depth+1)
            right_tree = self.build_tree(best_right_X, best_right_y, depth=depth+1)
            return Node(prediction=prediction, feature=best_feature, split=best_split, left_tree=left_tree, right_tree=right_tree)

        # if we did hit a leaf node
        return Node(prediction=prediction, feature=best_feature, split=best_split, left_tree=None, right_tree=None)

    def check_split(self, X, y, feature, split):
        '''
        check_split gets data corresponding to a split by using numpy indexing
        '''
        left_idx = np.where(X[:, feature] < split)
        right_idx = np.where(X[:, feature] >= split)
        left_x = X[left_idx]
        right_x = X[right_idx]
        left_y = y[left_idx]
        right_y = y[right_idx]

        # calculate gini impurity and gain for y, left_y, right_y
        gain = self.calculate_gini_gain(y, left_y, right_y)
        return gain, left_x, right_x, left_y, right_y

    def calculate_gini_gain(self, y, left_y, right_y):
        # not a leaf node
        # calculate gini impurity and gain
        gain = 0
        if len(left_y) > 0 and len(right_y) > 0:
            c_pos = sum(y)
            c_neg = len(y) - c_pos
            cl_pos = np.sum(left_y)
            cl_neg = len(left_y) - cl_pos
            cr_pos = np.sum(right_y)
            cr_neg = len(right_y) - cr_pos
            p_l = len(left_y) / len(y)
            p_r = len(right_y) / len(y)
            g_c = 1 - ((c_pos / len(y))**2) - ((c_neg / len(y))**2)
            g_l = 1 - ((cl_pos / len(left_y))**2) - ((cl_neg / len(left_y))**2)
            g_r = 1 - ((cr_pos / len(right_y))**2) - ((cr_neg /
                                                       len(right_y))**2)
            gain = g_c - p_l * g_l - p_r * g_r
            return gain
        # we hit leaf node
        # don't have any gain, and don't want to divide by 0
        return 0


class RandomForestClassifier():
    """
    Random Forest Classifier. Build a forest of decision trees.
    Use this forest for ensemble predictions

    YOU WILL NEED TO MODIFY THE DECISION TREE VERY SLIGHTLY TO HANDLE FEATURE
    BAGGING

    Parameters:
    -----------
    n_trees: int
        Number of trees in forest/ensemble
    max_features: int
        Maximum number of features to consider for a split when feature bagging
    max_depth: int
        Maximum depth of any decision tree in forest/ensemble
    """
    def __init__(self, n_trees, max_features, max_depth):
        self.n_trees = n_trees
        self.max_features = max_features
        self.max_depth = max_depth

        ##################
        # YOUR CODE HERE #
        ##################

    def fit(self, X, y):
        '''
        fit all trees
        '''
        bagged_X, bagged_y = self.bag_data(X, y)
        print('Fitting Random Forest...\n')
        self.trees = []
        for i in range(self.n_trees):
            print(i+1, end='\t\r')
            temp_tree = DecisionTreeClassifier(max_depth=self.max_depth,
                    max_features=self.max_features)
            temp_tree.fit(bagged_X[i], bagged_y[i])
            self.trees.append(temp_tree)

        print(len(self.trees))

    def bag_data(self, X, y, proportion=1.0):
        '''
        bag_data helper
        '''
        # Array of each tree's X data
        #   First index: tree index
        #   Second index: 2098 rows of randomly sampled data (repeats allowed)
        #   Third index: randomly sampled features on data (no repeats)
        bagged_X = []
        # array of prediction tags that match X data
        #   first index: tree index
        #   second index: 2098 randomly sampled tags that match X data sampling
        bagged_y = []
        for i in range(self.n_trees):
            # randomly sample indices of range of data (duplicates allowed)
            sampled_data = np.random.choice(len(X), 2098, replace=True)

            # use same index sampling to create matching data and tags
            data_sampled_X = [X[j] for j in sampled_data]
            data_sampled_y = [y[j] for j in sampled_data]

            # append data for current tree to total tree data
            bagged_X.append(data_sampled_X)
            bagged_y.append(data_sampled_y)

        # ensure data is still numpy arrays
        return np.array(bagged_X), np.array(bagged_y)


    def predict(self, X):
        '''
        predict
        '''
        preds = []

        for i in range(self.n_trees):
            tmp = self.trees[i].predict(X)
            tmp2 = [-1 if j == 0 else j for j in tmp]
            preds.append(tmp2)

        preds = np.sum(preds, axis=0)
        preds = [0 if j < 0 else 1 for j in preds]

        ##################
        # YOUR CODE HERE #
        ##################
        return preds


class AdaDecisionTreeClassifier():
    """
    Decision Tree Classifier. Class for building the decision tree and making
    predictions

    Parameters:
    ------------
    max_depth: int
        The maximum depth to build the tree. Root is at depth 0, a single split
        makes depth 1 (decision stump)
    """

    def __init__(self):
        self.max_depth = 1
        self.num_classes = 0
        self.root = None

    # take in features X and labels y
    # build a tree
    def fit(self, X, y, weights):
        '''
        fit a stump with the weights
        '''
        self.num_classes = len(set(y))
        self.root = self.build_tree(X, y, depth=1, weights)

    # make prediction for each example of features X
    def predict(self, X):
        preds = [self._predict(example) for example in X]

        return preds

    # prediction for a given example
    # traverse tree by following splits at nodes
    def _predict(self, example):
        node = self.root
        while node.left_tree:
            if example[node.feature] < node.split:
                node = node.left_tree
            else:
                node = node.right_tree
        return node.prediction

    # accuracy
    def accuracy_score(self, X, y):
        preds = self.predict(X)
        accuracy = (preds == y).sum()/len(y)
        return accuracy

    # function to build a decision tree
    def build_tree(self, X, y, depth, weights):
        num_samples, num_features = X.shape
        # which features we are considering for splitting on
        self.features_idx = np.arange(0, X.shape[1])

        # store data and information about best split
        # used when building subtrees recursively
        best_feature = None
        best_split = None
        best_gain = 0.0
        best_left_X = None
        best_left_y = None
        best_right_X = None
        best_right_y = None

        # what we would predict at this node if we had to
        # majority class
        num_samples_per_class = [np.sum(y == i) for i in range(self.num_classes)]
        prediction = np.argmax(num_samples_per_class)

        # if we haven't hit the maximum depth, keep building
        if depth <= self.max_depth:
            # consider each feature
            for feature in self.features_idx:
                # consider the set of all values for that feature to split on
                possible_splits = np.unique(X[:, feature])
                for split in possible_splits:
                    # get the gain and the data on each side of the split
                    # >= split goes on right, < goes on left
                    gain, left_X, right_X, left_y, right_y = self.check_split(X, y, feature, split, weights)
                    # if we have a better gain, use this split and keep track of data
                    if gain > best_gain:
                        best_gain = gain
                        best_feature = feature
                        best_split = split
                        best_left_X = left_X
                        best_right_X = right_X
                        best_left_y = left_y
                        best_right_y = right_y
        
        # if we haven't hit a leaf node
        # add subtrees recursively
        if best_gain > 0.0:
            left_tree = self.build_tree(best_left_X, best_left_y, depth=depth+1)
            right_tree = self.build_tree(best_right_X, best_right_y, depth=depth+1)
            return Node(prediction=prediction, feature=best_feature, split=best_split, left_tree=left_tree, right_tree=right_tree)

        # if we did hit a leaf node
        return Node(prediction=prediction, feature=best_feature, split=best_split, left_tree=None, right_tree=None)

    def check_split(self, X, y, feature, split, weights):
        '''
        check_split gets data corresponding to a split by using numpy indexing
        '''
        left_idx = np.where(X[:, feature] < split)
        right_idx = np.where(X[:, feature] >= split)
        left_x = X[left_idx]
        right_x = X[right_idx]
        left_y = y[left_idx]
        right_y = y[right_idx]

        # calculate gini impurity and gain for y, left_y, right_y
        gain = self.calculate_gini_gain(y, left_y, right_y)
        return gain, left_x, right_x, left_y, right_y

    def calculate_gini_gain(self, y, left_y, right_y, weights):
        # not a leaf node
        # calculate gini impurity and gain
        gain = 0
        if len(left_y) > 0 and len(right_y) > 0:
            c_pos = sum(y)
            c_neg = len(y) - c_pos
            cl_pos = np.sum(left_y)
            cl_neg = len(left_y) - cl_pos
            cr_pos = np.sum(right_y)
            cr_neg = len(right_y) - cr_pos
            p_l = len(left_y) / len(y)
            p_r = len(right_y) / len(y)
            g_c = 1 - ((c_pos / len(y))**2) - ((c_neg / len(y))**2)
            g_l = 1 - ((cl_pos / len(left_y))**2) - ((cl_neg / len(left_y))**2)
            g_r = 1 - ((cr_pos / len(right_y))**2) - ((cr_neg /
                                                       len(right_y))**2)
            gain = g_c - p_l * g_l - p_r * g_r
            return gain
        # we hit leaf node
        # don't have any gain, and don't want to divide by 0
        return 0

################################################
# YOUR CODE GOES IN ADABOOSTCLASSIFIER         #
# MUST MODIFY THIS EXISTING DECISION TREE CODE #
################################################
class AdaBoostClassifier():
    def __init__(self, L):
        self.number_of_trees = L

    def fit(self, x, y):
        y[y == 0] = -1
        d = [1/x.shape[0] for n in range(x.shape[0])]
        for t in range(self.number_of_trees):
            tree = AdaDecisionTreeClassifier()
            tree.fit(x, y, d)
            e = self.error(tree, x, y, d)

            alpha = self.alpha(e)
            d_t = self.update_weights(e, alpha, tree, x, y)
            d = self.normalize(d_t)


