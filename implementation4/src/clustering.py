import numpy as np


class KMeans():
    """
    KMeans. Class for building an unsupervised clustering model
    """

    def __init__(self, k, max_iter=20):

        """
        :param k: the number of clusters
        :param max_iter: maximum number of iterations
        """

        self.k = k
        self.max_iter = max_iter

    def init_center(self, x, param_num=-1):
        """
        init_center initializes the center of the clusters using the given
        input
        :param x: input of shape (n, m)
        :return: updates the self.centers
        """

        self.centers = np.zeros((self.k, x.shape[1]))

        if param_num > 0:
            print(param_num)
            exit()
            idx = np.random.choice(param_num, self.k, replace=False)
        else:
            idx = np.random.choice(x.shape[1], self.k, replace=False)

        for i, index in enumerate(idx):
            self.centers[i] = x[index]

        # print("Test center: ", self.centers[0])

    def revise_centers(self, x, labels):
        """
        it updates the centers based on the labels
        :param x: the input data of (n, m)
        :param labels: the labels of (n, ). Each labels[i] is the cluster index of sample x[i]
        :return: updates the self.centers
        """

        for i in range(self.k):
            wherei = np.squeeze(np.argwhere(labels == i), axis=1)
            #if x[wherei, :].size == 0:
            #    continue
            self.centers[i, :] = x[wherei, :].mean(0)

    def predict(self, x):
        """
        returns the labels of the input x based on the current self.centers
        :param x: input of (n, m)
        :return: labels of (n,). Each labels[i] is the cluster index for sample x[i]
        """
        labels = np.zeros((x.shape[0]), dtype=int)
        ##################################
        #      YOUR CODE GOES HERE       #
        ##################################
        
        # For every sample, assign it to a center
        for i, sample in enumerate(x):
            min_d = float('inf')
            min_c = -1
            # This could probably be made to be a matrix math op, but nested
            # for loop go brrr
            for j, center in enumerate(self.centers):
                dist = np.linalg.norm(sample - center)

                if dist < min_d:
                    min_d = dist
                    min_c = j
            labels[i] = min_c
        return labels

    def get_sse(self, x, labels):
        """
        for a given input x and its cluster labels, it computes the sse with
        respect to self.centers
        :param x:  input of (n, m)
        :param labels: label of (n,)
        :return: float scalar of sse
        """

        ##################################
        #      YOUR CODE GOES HERE       #
        ##################################
        # Dist from the center for a given sample
        sse = 0.
        for i, sample in enumerate(x):
            sse += np.linalg.norm(sample - self.centers[labels[i]])

        return sse

    def get_purity(self, x, y):
        """
        computes the purity of the labels (predictions) given on x by the model
        :param x: the input of (n, m)
        :param y: the ground truth class labels
        :return:
        """
        labels = self.predict(x)
        purity = 0
        ##################################
        #      YOUR CODE GOES HERE       #
        ##################################
        
        # Prep the centers to have tags associated
        tags = np.zeros(self.k)
        sums = np.zeros((self.k, 6))
        for i, label in enumerate(labels):
            sums[label][y[i]-1] += 1

        # Set the largest tag to be one for the center
        for i, sum_ in enumerate(sums):
            tags[i] = np.argmax(sum_)+1

        # Calculate percent correct
        correct = 0
        for label, truth in zip(labels, y):
            if tags[label] == truth:
                correct += 1

        purity = correct / len(y)
        print(purity)
        return purity

    def fit(self, x, param_num=-1):
        """
        this function iteratively fits data x into k-means model. The result of
        the iteration is the cluster centers.
        :param x: input data of (n, m)
        :return: computes self.centers. It also returns sse_veersus_iterations
        for x.
        """

        # intialize self.centers
        self.init_center(x, param_num=-1)

        sse_vs_iter = []
        for iter in range(self.max_iter):
            # finds the cluster index for each x[i] based on current centers
            labels = self.predict(x)

            # revises the values of self.centers based on the x and current labels
            self.revise_centers(x, labels)

            # computes the sse based on the current labels and centers.
            sse = self.get_sse(x, labels)

            sse_vs_iter.append(sse)

        return sse_vs_iter
