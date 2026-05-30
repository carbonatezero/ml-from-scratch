import numpy as np

from src.models.knn import KNearestNeighbor


def test_knn_distance_implementations_agree():
    X_train = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y_train = np.array([0, 1, 2])
    X_test = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])

    knn = KNearestNeighbor()
    knn.train(X_train, y_train)

    dists_two = knn.compute_distances_two_loops(X_test)
    dists_one = knn.compute_distances_one_loop(X_test)
    dists_no = knn.compute_distances_no_loops(X_test)

    assert np.allclose(dists_two, dists_one)
    assert np.allclose(dists_two, dists_no)


def test_knn_predict_labels_breaks_ties_with_smaller_label():
    knn = KNearestNeighbor()
    knn.train(np.array([[0.0], [2.0]]), np.array([1, 0]))

    y_pred = knn.predict(np.array([[1.0]]), k=2, num_loops=0)

    assert np.array_equal(y_pred, np.array([0.0]))
