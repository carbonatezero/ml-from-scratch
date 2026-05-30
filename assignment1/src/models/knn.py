from builtins import range
from builtins import object
import numpy as np
from past.builtins import xrange


class KNearestNeighbor(object):
    """ a kNN classifier with L2 distance """

    def __init__(self):
        pass

    def train(self, X, y):
        """
        Train the classifier. For k-nearest neighbors this is just
        memorizing the training data.

        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data
          consisting of num_train samples each of dimension D.
        - y: A numpy array of shape (N,) containing the training labels, where
             y[i] is the label for X[i].
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, num_loops=0):
        """
        Predict labels for test data using this classifier.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data consisting
             of num_test samples each of dimension D.
        - k: The number of nearest neighbors that vote for the predicted labels.
        - num_loops: Determines which implementation to use to compute distances
          between training points and testing points.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError("Invalid value %d for num_loops" % num_loops)

        return self.predict_labels(dists, k=k)

    def compute_distances_two_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a nested loop over both the training data and the
        test data.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data.

        Returns:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          is the Euclidean distance between the ith test point and the jth training
          point.
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                #####################################################################
                # TODO:                                                             #
                # Compute the l2 distance between the ith test point and the jth    #
                # training point, and store the result in dists[i, j]. You should   #
                # not use a loop over dimension, nor use np.linalg.norm().          #
                #####################################################################
                
                diff = X[i] - self.X_train[j]
                dists[i, j] = np.sqrt(np.sum(diff * diff))

                # pass
        return dists

    def compute_distances_one_loop(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a single loop over the test data.

        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            diff = self.X_train - X[i]
            dists[i, :] = np.sqrt(np.sum(diff * diff, axis=1))
        return dists

    def compute_distances_no_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using no explicit loops.

        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        #########################################################################
        # TODO:                                                                 #
        # Compute the l2 distance between all test points and all training      #
        # points without using any explicit loops, and store the result in      #
        # dists.                                                                #
        #                                                                       #
        # You should implement this function using only basic array operations; #
        # in particular you should not use functions from scipy,                #
        # nor use np.linalg.norm().                                             #
        #                                                                       #
        # HINT: Try to formulate the l2 distance using matrix multiplication    #
        #       and two broadcast sums.                                         #
        #########################################################################

        train = self.X_train 
        X_squared = np.sum(X * X, axis=1, keepdims=True) 
        train_squared = np.sum(train * train, axis=1) 
        cross_term = X @ train.T 
        dists = np.sqrt(np.maximum(X_squared + train_squared - 2 * cross_term, 0.0))

        return dists

    def predict_labels(self, dists, k=1):
        """
        Given a matrix of distances between test points and training points,
        predict a label for each test point.

        Inputs:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          gives the distance betwen the ith test point and the jth training point.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            # A list of length k storing the labels of the k nearest neighbors to
            # the ith test point.
            closest_y = []
            #########################################################################
            # TODO:                                                                 #
            # Use the distance matrix to find the k nearest neighbors of the ith    #
            # testing point, and use self.y_train to find the labels of these       #
            # neighbors. Store these labels in closest_y.                           #
            # Hint: Look up the function numpy.argsort.                             #
            #########################################################################
            sorted_idx = np.argsort(dists[i])
            knn_idx = sorted_idx[:k]
            closest_y = self.y_train[knn_idx]


            #########################################################################
            # TODO:                                                                 #
            # Now that you have found the labels of the k nearest neighbors, you    #
            # need to find the most common label in the list closest_y of labels.   #
            # Store this label in y_pred[i]. Break ties by choosing the smaller     #
            # label.                                                                #
            #########################################################################
            labels, counts = np.unique(closest_y, return_counts=True)
            max_count = np.max(counts)
            candidates = labels[counts == max_count]
            y_pred[i] = np.min(candidates)


        return y_pred

def main():
    knn = KNearestNeighbor()

    X_train = np.array([[1, 2], [3, 4], [5, 6]])
    y_train = np.array([0, 1, 2])
    knn.train(X_train, y_train)

    X_test = np.array([[1, 2], [2, 3], [3, 4]])

    # Compute distances using all implementations
    dists_two = knn.compute_distances_two_loops(X_test)
    dists_one = knn.compute_distances_one_loop(X_test)
    dists_no  = knn.compute_distances_no_loops(X_test)

    # Reference distances using np.linalg.norm
    num_test = X_test.shape[0]
    num_train = X_train.shape[0]
    dists_ref = np.zeros((num_test, num_train))
    for i in range(num_test):
        for j in range(num_train):
            dists_ref[i, j] = np.linalg.norm(X_test[i] - X_train[j])

    # Compare all distance implementations to reference
    if not np.allclose(dists_two, dists_ref):
        print("Two-loop version failed (distances).")
        return

    if not np.allclose(dists_one, dists_ref):
        print("One-loop version failed (distances).")
        return

    if not np.allclose(dists_no, dists_ref):
        print("No-loop version failed (distances).")
        return

    # Cross-compare implementations
    if not (np.allclose(dists_two, dists_one) and np.allclose(dists_two, dists_no)):
        print("Implementations do not match each other (distances).")
        return

    print("All distance implementations passed sanity check.")

    # ---------------------------
    # Now sanity-check prediction
    # ---------------------------

    # Expected predictions for k=1:
    # X_test[0]=[1,2] nearest to train[0] label 0
    # X_test[1]=[2,3] tie between train[0] and train[1] (same distance); choose smaller label -> 0
    # X_test[2]=[3,4] nearest to train[1] label 1
    y_expected_k1 = np.array([0, 0, 1])

    # Test predict() (which should call one of the distance implementations internally)
    y_pred_k1 = knn.predict(X_test, k=1, num_loops=0)
    if not np.array_equal(y_pred_k1, y_expected_k1):
        print("predict() failed for k=1, num_loops=0.")
        print("Expected:", y_expected_k1)
        print("Got:     ", y_pred_k1)
        return

    # Also test predict_labels() directly using the precomputed distance matrices
    y_pred_two = knn.predict_labels(dists_two, k=1)
    y_pred_one = knn.predict_labels(dists_one, k=1)
    y_pred_no  = knn.predict_labels(dists_no,  k=1)

    if not (np.array_equal(y_pred_two, y_expected_k1) and
            np.array_equal(y_pred_one, y_expected_k1) and
            np.array_equal(y_pred_no,  y_expected_k1)):
        print("predict_labels() failed for k=1 on one or more distance matrices.")
        print("Expected:", y_expected_k1)
        print("Got two: ", y_pred_two)
        print("Got one: ", y_pred_one)
        print("Got no:  ", y_pred_no)
        return

    # Optional: check a tie-breaking scenario for k=2 on the middle test point [2,3]
    # For i=1, the two nearest are train[0] label 0 and train[1] label 1 -> tie -> choose 0
    y_expected_k2 = np.array([0, 0, 0])
    y_pred_k2 = knn.predict(X_test, k=2, num_loops=0)
    if not np.array_equal(y_pred_k2, y_expected_k2):
        print("predict() failed for k=2 tie-break check.")
        print("Expected:", y_expected_k2)
        print("Got:     ", y_pred_k2)
        return

    print("Prediction sanity checks passed (predict and predict_labels).")


if __name__ == "__main__":
    main()