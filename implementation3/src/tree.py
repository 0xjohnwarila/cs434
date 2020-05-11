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
            # Get the center counts
            c_pos = sum(y)
            c_neg = len(y) - c_pos

            # Get left counts
            cl_pos = np.sum(left_y)
            cl_neg = len(left_y) - cl_pos

            # Get right counts
            cr_pos = np.sum(right_y)
            cr_neg = len(right_y) - cr_pos

            # Priors
            p_l = len(left_y) / len(y)
            p_r = len(right_y) / len(y)

            # U(A)
            g_c = 1 - ((c_pos / len(y))**2) - ((c_neg / len(y))**2)
            
            # U(AL)
            g_l = 1 - ((cl_pos / len(left_y))**2) - ((cl_neg / len(left_y))**2)

            # U(AR)
            g_r = 1 - ((cr_pos / len(right_y))**2) - ((cr_neg /
                                                       len(right_y))**2)
            # B = U(A) - pl*U(AL) - pr*U(AR)
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

    def fit(self, X, y):
        '''
        fit all trees
        '''

        # get the bagged x and y
        bagged_X, bagged_y = self.bag_data(X, y)
        print('Fitting Random Forest...\n')
        self.trees = []
        # Fit each tree
        for i in range(self.n_trees):
            print(i+1, end='\t\r')
            temp_tree = DecisionTreeClassifier(max_depth=self.max_depth,
                    max_features=self.max_features)
            temp_tree.fit(bagged_X[i], bagged_y[i])
            self.trees.append(temp_tree)

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

        # For all our trees, take a vote
        for i in range(self.n_trees):
            tmp = self.trees[i].predict(X)
            # List comprehension of the prediction
            tmp2 = [-1 if j == 0 else j for j in tmp]
            preds.append(tmp2)

        # Sum the votes
        preds = np.sum(preds, axis=0)
        preds = [0 if j < 0 else 1 for j in preds]

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
        self.root = self.build_tree(X, y, weights, depth=1)

    # make prediction for each example of features X
    def predict(self, X):
        preds = [self._predict(example) for example in X]

        return preds

    # prediction for a given example
    # traverse tree by following splits at nodes
    def _predict(self, example):

        # Special prediction expecting there only to be a left and right node, storing the predictions
        node = self.root
        if example[node.feature] < node.split:
            return node.left_tree.prediction
        return node.right_tree.prediction

    # accuracy
    def accuracy_score(self, X, y):
        preds = self.predict(X)
        accuracy = (preds == y).sum()/len(y)
        return accuracy

    # function to build a decision tree
    def build_tree(self, X, y, weights, depth):
        num_samples, num_features = X.shape
        # which features we are considering for splitting on
        self.features_idx = np.arange(0, X.shape[1])

        # store data and information about best split
        # used when building subtrees recursively
        best_feature = None
        best_split = None
        best_error = float('inf')
        # Prediction left
        p_l = None
        # Prediction right
        p_r = None



        # if we haven't hit the maximum depth, keep building
        if depth <= self.max_depth:
            # consider each feature
            for feature in self.features_idx:
                # consider the set of all values for that feature to split on
                possible_splits = np.unique(X[:, feature])
                for split in possible_splits:
                    # >= split goes on right, < goes on left
                    error, prediction_l, prediction_r = self.check_split(X, y, feature, split, weights)
                    # If the error is lower, remember it
                    if error < best_error:
                        best_error = error
                        best_feature = feature
                        best_split = split
                        p_l = prediction_l
                        p_r = prediction_r
        
        # Store the predictions in the left and right nodes
        left_tree = Node(p_l, None, None, None, None)
        right_tree = Node(p_r, None, None, None, None)

        # Return the stump
        return Node(prediction=None, feature=best_feature, split=best_split, left_tree=left_tree, right_tree=right_tree)

    def check_split(self, X, y, feature, split, weights):
        '''
        check_split gets data corresponding to a split by using numpy indexing.
        Also gets the prediction and error for both sides
        '''
        weights = np.array(weights)

        left_idx = np.where(X[:, feature] < split)
        right_idx = np.where(X[:, feature] >= split)

        left_y = y[left_idx]
        right_y = y[right_idx]

        left_w = weights[left_idx]
        right_w = weights[right_idx]

        # Get the weights for both sides
        left_w_pos = 0
        left_w_neg = 0
        for tag, w in zip(left_y, left_w):
            if tag > 0:
                left_w_pos += w
            else:
                left_w_neg += w

        right_w_pos = 0
        right_w_neg = 0

        for tag, w in zip(right_y, right_w):
            if tag > 0:
                right_w_pos += w
            else:
                right_w_neg += w


        # Get the prediction for the left side
        p_l = -1
        if left_w_pos > left_w_neg:
            p_l = 1

        # Get the prediction for the right side
        p_r = -1
        if right_w_pos > right_w_neg:
            p_r = 1
        
        # Calculate the error for the split
        error = 0
        for sample, tag, weight in zip(X, y, weights):
            pred = 0
            if sample[feature] < split:
                pred = p_l
            else:
                pred = p_r
            tmp = 1
            if pred == tag:
                tmp = 0
            error += weight * tmp
            
        # Return the error and the predictions
        return error, p_l, p_r

class AdaBoostClassifier():
    def __init__(self, L):
        self.number_of_trees = L

    def fit(self, x, y):
        self.trees = []

        # Convert the labels to -1, 1 format
        y[y == 0] = -1
        # Get the base weights
        d = [1/x.shape[0] for n in range(x.shape[0])]

        # Train every stump
        for t in range(self.number_of_trees):
            tree = AdaDecisionTreeClassifier()
            # Fit the tree on the data and the weights D
            tree.fit(x, y, d)
            self.trees.append(tree)
            # Calculate the weighted error of the tree
            e = self._error(tree, x, y, d)
            # Calculate alpha from the error
            alpha = self._alpha(e)
            # Update and Normalize the weights
            d_t = self._update_weights(alpha, tree, x, y, d)
            d = self._normalize(d_t)

    def predict(self, x):
        predictions = []
        # Get the predictions from each tree
        for tree in self.trees:
            temp = tree.predict(x)
            predictions.append(tree.predict(x))

        # Sum and take the vote
        predictions = np.sum(predictions, axis=0)
        predictions = [0 if i < 0 else 1 for i in predictions]
        return predictions

    def _update_weights(self, alpha, tree, x, y, d):
        # Update every weight based off alpha
        i = 0
        for sample, tag in zip(x, y):
            prediction = tree._predict(sample)
            if prediction == tag:
                d[i] = d[i]*np.exp(-alpha)
            else:
                d[i] = d[i]*np.exp(alpha)

            i += 1

        return d

    def _normalize(self, d):
        # Normalize the weights to sum to 1
        sum = np.sum(np.array(d))
        d = [i/sum for i in d]
        return d

    def _error(self, tree, x, y, d):
        error = 0
        # Calculate the weighted error.
        for sample, tag, weight in zip(x, y, d):
            tmp = 1
            if tree._predict(sample) == tag:
                tmp = 0
            # When the tree makes an error, add the weight for that sample
            error += weight * tmp
        return error

    def _alpha(self, e):
        # Caclualte alpha. 1/2 * ln((1-e)/e)
        partial = (1 - e) / e
        return .5 * np.log(partial)
