import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn import manifold, decomposition
import pickle
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import normalize

def gaussian_example():
    X, y_true = make_blobs(n_samples=200, centers=4,
                           cluster_std=0.60, random_state=0)

    # plt.plot(X[:, 0], X[:, 1], '.k')
    # plt.show()
    return X


def circles_example():
    """
    an example function for generating and plotting synthetic data.
    """

    t = np.arange(0, 2 * np.pi, 0.03)
    length = np.shape(t)
    length = length[0]
    circle1 = np.matrix([np.cos(t) + 0.1 * np.random.randn(length),
                         np.sin(t) + 0.1 * np.random.randn(length)])
    circle2 = np.matrix([2 * np.cos(t) + 0.1 * np.random.randn(length),
                         2 * np.sin(t) + 0.1 * np.random.randn(length)])
    circle3 = np.matrix([3 * np.cos(t) + 0.1 * np.random.randn(length),
                         3 * np.sin(t) + 0.1 * np.random.randn(length)])
    circle4 = np.matrix([4 * np.cos(t) + 0.1 * np.random.randn(length),
                         4 * np.sin(t) + 0.1 * np.random.randn(length)])
    circles = np.concatenate((circle1, circle2, circle3, circle4), axis=1)

    # plt.plot(circles[0,:], circles[1,:], '.k')
    # plt.show()

    return circles

def apml_pic_example(path='APML_pic.pickle'):
    """
    an example function for loading and plotting the APML_pic data.

    :param path: the path to the APML_pic.pickle file
    """

    with open(path, 'rb') as f:
        apml = pickle.load(f)

    # plt.plot(apml[:, 0], apml[:, 1], '.')
    # plt.show()

    return apml


def microarray_exploration(data_path='microarray_data.pickle',
                            genes_path='microarray_genes.pickle',
                            conds_path='microarray_conds.pickle'):
    """
    an example function for loading and plotting the microarray data.
    Specific explanations of the data set are written in comments inside the
    function.

    :param data_path: path to the data file.
    :param genes_path: path to the genes file.
    :param conds_path: path to the conds file.
    """

    # loading the data:
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    with open(genes_path, 'rb') as f:
        genes = pickle.load(f)
    with open(conds_path, 'rb') as f:
        conds = pickle.load(f)

    # look at a single value of the data matrix:
    i = 128
    j = 63
    print("gene: " + str(genes[i]) + ", cond: " + str(
        conds[0, j]) + ", value: " + str(data[i, j]))

    # make a single bar plot:
    plt.figure()
    plt.bar(np.arange(len(data[i, :])) + 1, data[i, :])
    plt.show()

    # look at two conditions and their correlation:
    plt.figure()
    plt.scatter(data[:, 27], data[:, 29])
    plt.plot([-5,5],[-5,5],'r')
    plt.show()

    # see correlations between conds:
    correlation = np.corrcoef(np.transpose(data))
    plt.figure()
    plt.imshow(correlation, extent=[0, 1, 0, 1])
    plt.colorbar()
    plt.show()

    # look at the entire data set:
    plt.figure()
    plt.imshow(data, extent=[0, 1, 0, 1], cmap="hot", vmin=-3, vmax=3)
    plt.colorbar()
    plt.show()


def euclid(X, Y):
    """
    return the pair-wise euclidean distance between two data matrices.
    :param X: NxD matrix.
    :param Y: MxD matrix.
    :return: NxM euclidean distance matrix.
    """

    return euclidean_distances(X, Y)


def euclidean_centroid(X):
    """
    return the center of mass of data points of X.
    :param X: a sub-matrix of the NxD data matrix that defines a cluster.
    :return: the centroid of the cluster.
    """

    return np.sum(X, axis=0, keepdims=True) / X.shape[0]


def kmeans_pp_init(X, K, metric):
    """
    The initialization function of kmeans++, returning k centroids.
    :param X: A nXd data matrix.
    :param k: The number of clusters.
    :param metric: a metric function like specified in the kmeans documentation.
    :return: kxD matrix with rows containing the centroids.
    """

    n, d = X.shape
    mu = np.zeros([K, d])
    mu[0, :] = X[np.random.randint(0, n - 1, 1), :]

    for k in range(K - 1):
        D = metric(X, mu[:(k + 1), :])
        D_square = np.min(D, axis=1) ** 2
        D_square_normalized = D_square / np.sum(D_square)
        w = np.random.choice(np.arange(n), p=D_square_normalized)
        mu[k + 1, :] = X[[w], :]
    return mu


def kmeans(X, K, iterations=10, metric=euclid, center=euclidean_centroid, init=kmeans_pp_init):
    """
    The K-Means function, clustering the data X into k clusters.
    :param X: A nXd data matrix.
    :param K: The number of desired clusters.
    :param iterations: The number of iterations.
    :param metric: A function that accepts two data matrices and returns their
            pair-wise distance. For a NxD and KxD matrices for instance, return
            a NxK distance matrix.
    :param center: A function that accepts a sub-matrix of X where the rows are
            points in a cluster, and returns the cluster centroid.
    :param init: A function that accepts a data matrix and k, and returns k initial centroids.
    :return: a tuple of (clustering, centroids)
    clustering - A N-dimensional vector with indices from 0 to k-1, defining the clusters.
    centroids - The kxD centroid matrix.
    """

    n, d = X.shape

    track_loss = []
    track_convergence = []
    track_centroids = []
    track_clustering = []
    for i in range(iterations):
        loss = []
        centroids = init(X, K, metric)
        # repeat till convergence
        while True:
            # update clusters
            dists = np.zeros([n, K])
            for k in range(K):
                dists[:, [k]] = np.linalg.norm((X - centroids[[k], :]), ord=2, axis=1, keepdims=True)
            clustering = np.argmin(dists, axis=1)

            # update means
            for k in range(K):
                centroids[k, :] = center(X[clustering == k, :])

            # MSE loss calculation
            iter_loss = 0
            for k in range(K):
                iter_loss += np.sum((X[clustering == k, :] - centroids[[k], :]) ** 2)

            loss.append(iter_loss)

            if len(loss) > 1 and np.abs(loss[-1]-loss[-2]) < 1e-5:
                break

        track_convergence.append(loss[-1])
        track_loss.append(loss)
        track_clustering.append(clustering)
        track_centroids.append(centroids)

    best_res = np.argmin(np.array(track_convergence))
    clustering = np.array(track_clustering[best_res])
    centroids = np.array(track_centroids[best_res])

    return clustering, centroids, track_convergence[best_res], track_loss[best_res]


def gaussian_kernel(X, sigma):
    """
    calculate the gaussian kernel similarity of the given distance matrix.
    :param X: A NxN distance matrix.
    :param sigma: The width of the Gaussian in the heat kernel.
    :return: NxN similarity matrix.
    """

    return np.exp(- (euclid(X, X) ** 2) / (2 * sigma ** 2))


def mnn(X, m):
    """
    calculate the m nearest neighbors similarity of the given distance matrix.
    :param X: A NxN distance matrix.
    :param m: The number of nearest neighbors.
    :return: NxN similarity matrix.
    """

    # get the m nearest neighbors indexes
    n, d = X.shape
    D = np.argsort(euclid(X, X), axis=1)[:, :m]
    W = np.zeros((n, n), dtype=bool)
    for i in range(n):
        W[D[i, :], i] = True
        W[i, D[i, :]] = True
    return W


def spectral(X, K, similarity_param, similarity):
    """
    Cluster the data into k clusters using the spectral clustering algorithm.
    :param X: A nXd data matrix.
    :param k: The number of desired clusters.
    :param similarity_param: m for mnn, sigma for the Gaussian kernel.
    :param similarity: The similarity transformation of the data.
    :return: clustering, as in the kmeans implementation.
    """

    n, d = X.shape
    W = similarity(X, similarity_param)
    D = np.diag(np.sum(W, axis=1) ** (-0.5))
    L = np.identity(n) - D @ W @ D
    w, v = np.linalg.eigh(L)
    # way to find the spectral gap
    if K is None:
        div = np.diff(w)
        div_roll = np.roll(div, 1)
        div_roll[0] = 1
        eigen_gap_plot = div/div_roll
        plt.figure()
        K = np.argmax(eigen_gap_plot > 100) + 1
        plt.plot(eigen_gap_plot[:K+1])
        plt.title("The chosen K is {}".format(K))
        plt.show()


    k_min_eigenvectors = v[:, np.argsort(w)[:K]]
    k_min_eigenvectors_normalized = normalize(k_min_eigenvectors, norm='l2')

    return kmeans(k_min_eigenvectors_normalized, K)


def elbow(X, ks, name, clustering_func, *args):
    losses = []
    for K in range(1, ks):
        clustering, centroids, loss, _ = clustering_func(X, K, *args)
        losses.append(loss)
    plt.figure()
    plt.plot(np.arange(1, ks), losses, 'k.', MarkerSize=15)
    plt.title("Elbow plot to choose K on {} data".format(name))
    plt.show()
    return losses


def silhouette(X, ks, clustering_func, *args):
    silhouettes = []
    S_k = []
    dists = euclid(X, X)
    n, d = X.shape
    for K in range(2, ks):
        clustering, centroids, loss, _ = clustering_func(X, K, *args)
        cluster_indexes = []
        cluster_sizes = np.zeros(K)
        for k in range(K):
            C = clustering == k
            cluster_indexes.append(C)
            cluster_sizes[k] = np.sum(C)

        silhouette_val = np.zeros(n)
        for i in range(n):
            sum_dists = np.zeros(K)
            for k in range(K):
                sum_dists[k] = np.sum(dists[i, cluster_indexes[k]])
            a_i = sum_dists[clustering[i]]/(cluster_sizes[clustering[i]] - 1)
            other_cluster = np.ones(sum_dists.shape, dtype=bool)
            other_cluster[clustering[i]] = False
            b_i = np.min(sum_dists[other_cluster]/cluster_sizes[other_cluster])

            silhouette_val[i] = (b_i - a_i) / max(a_i, b_i)
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        silhouette_val_sorted = []
        for k in range(K):
            silhouette_val_sorted = np.concatenate([silhouette_val_sorted, np.sort(silhouette_val[clustering == k])])
        silhouettes.append(silhouette_val_sorted)
        S_k.append(np.sum(silhouette_val_sorted)/n)
    return silhouettes, S_k


def estimate_sigma(X, percentile, name):
    S = euclid(X, X)
    hist, bins = np.histogram(S, density=True, bins=100)
    c_hist = np.cumsum(hist / np.sum(hist))
    plt.figure()
    plt.plot(c_hist)
    plt.title("Sigma estimation - histogram of distances for {} data".format(name))
    plt.show()
    return bins[np.argmax(c_hist > percentile/100) - 1]


def plot_silhouette(silhouettes, S_k, k, name):
    plt.figure()
    plt.fill_betweenx(np.arange(0, silhouettes.shape[0], 1), 0, silhouettes)
    plt.title("Silhouette plot for K={} on {} data".format(k,name))
    plt.show()


def clustering_plot(X, clustering, centroids, losses, name):
    K = np.max(clustering) + 1
    plt.figure()
    for k in range(K):
        indexes = (clustering == k)
        plt.plot(X[indexes, 0], X[indexes, 1], '.', MarkerSize=10)
        plt.plot(centroids[k, 0], centroids[k, 1], 'r+', mew=10, ms=20)
        plt.plot(centroids[k, 0], centroids[k, 1], 'k+', mew=5, ms=15)
    plt.title(name + " with K={}".format(K))
    plt.show()
    plt.figure()
    plt.plot(losses, 'k.', MarkerSize=15)
    plt.title("loss function for {}".format(name))
    plt.show()


def compare_sim_matrix(data, clustering, similarity_param, similarity, name):
    clustered = data[clustering == 0, :]
    for k in range(1, 4):
        clustered = np.vstack((clustered, data[clustering == k, :]))
    W_orig = similarity(data, similarity_param)
    plt.imshow(W_orig)
    plt.title(name + " shuffled")
    plt.figure()
    W = similarity(clustered, similarity_param)
    plt.imshow(W)
    plt.title(name + " clustered")
    plt.show()


def kmeans_pp_synthetic(circles, apml, gauss):
    circles_clustering, circles_centroids, circles_loss, circles_losses = kmeans(circles,
                                                                                 K=4,
                                                                                 iterations=10,
                                                                                 metric=euclid,
                                                                                 center=euclidean_centroid,
                                                                                 init=kmeans_pp_init)
    clustering_plot(circles, circles_clustering, circles_centroids, circles_losses,
                    "K-Mean++ clustering on circles data")

    apml_clustering, apml_centroids, apml_loss, apml_losses = kmeans(apml,
                                                                     K=8,
                                                                     iterations=10,
                                                                     metric=euclid,
                                                                     center=euclidean_centroid,
                                                                     init=kmeans_pp_init)
    clustering_plot(apml, apml_clustering, apml_centroids, apml_losses, "K-Mean++ clustering on apml data")

    gauss_clustering, gauss_centroids, gauss_loss, gauss_losses = kmeans(gauss,
                                                                         K=4,
                                                                         iterations=10,
                                                                         metric=euclid,
                                                                         center=euclidean_centroid,
                                                                         init=kmeans_pp_init)
    clustering_plot(gauss, gauss_clustering, gauss_centroids, gauss_losses, "K-Mean++ clustering on gaussian data")


def spectral_synthetic_gauss(circles, apml, gauss, compare=False):
    circles_clustering, circles_centroids, circles_loss, circles_losses = spectral(circles,
                                                                                   K=4,
                                                                                   similarity_param=estimate_sigma(
                                                                                       circles, 1, 'circles'),
                                                                                   similarity=gaussian_kernel)
    if compare:
        compare_sim_matrix(circles, circles_clustering, estimate_sigma(circles, 1, "circles"), gaussian_kernel, "Similarity Matrix Circle (heat kernel)")
    clustering_plot(circles, circles_clustering, circles_centroids, circles_losses, "Spectral clustering(heat kernel) on circles data")

    apml_clustering, apml_centroids, apml_loss, apml_losses = spectral(apml,
                                                                       K=9,
                                                                       similarity_param=estimate_sigma(
                                                                           apml, 1, 'apml'),
                                                                       similarity=gaussian_kernel)
    if compare:
        compare_sim_matrix(apml, apml_clustering, estimate_sigma(apml, 2, 'apml'), gaussian_kernel, "Similarity Matrix APML(heat kernel) ")

    clustering_plot(apml, apml_clustering, apml_centroids, apml_losses, "Spectral clustering(heat kernel) on apml data")

    gauss_clustering, gauss_centroids, gauss_loss, gauss_losses = spectral(gauss,
                                                                           K=4,
                                                                           similarity_param=estimate_sigma(
                                                                               gauss, 2, 'gaussian'),
                                                                           similarity=gaussian_kernel)
    if compare:
        compare_sim_matrix(gauss, gauss_clustering, estimate_sigma(gauss, 2, 'gaussian'), gaussian_kernel, "Similarity Matrix gaussian (heat kernel)")
    clustering_plot(gauss, gauss_clustering, gauss_centroids, gauss_losses, "Spectral clustering(heat kernel) on gaussian data")


def spectral_synthetic_mnn(circles, apml, gauss, compare=False):
    circles_clustering, circles_centroids, circles_loss, circles_losses = spectral(circles,
                                                                                   K=4,
                                                                                   similarity_param=10,
                                                                                   similarity=mnn)
    if compare:
        compare_sim_matrix(circles, circles_clustering, 10, mnn, "Similarity Matrix circles (mnn)")
    clustering_plot(circles, circles_clustering, circles_centroids, circles_losses, "Spectral clustering(mnn) on circles data")

    apml_clustering, apml_centroids, apml_loss, apml_losses = spectral(apml,
                                                                       K=9,
                                                                       similarity_param=15,
                                                                       similarity=mnn)
    if compare:
        compare_sim_matrix(apml, apml_clustering, 15, mnn, "Similarity Matrix APML (mnn)")
    clustering_plot(apml, apml_clustering, apml_centroids, apml_losses, "Spectral clustering(mnn) on apml data")


    gauss_clustering, gauss_centroids, gauss_loss, gauss_losses = spectral(gauss,
                                                                           K=4,
                                                                           similarity_param=10,
                                                                           similarity=mnn)
    if compare:
        compare_sim_matrix(gauss, gauss_clustering, 10, mnn, "Similarity Matrix gaussian (mnn)")
    clustering_plot(gauss, gauss_clustering, gauss_centroids, gauss_losses, "Spectral clustering(mnn) on gaussian data")


def elbow_losses(circles, apml, gauss):
    elbow(circles, 10, "circles", kmeans, 10, euclid, euclidean_centroid, kmeans_pp_init)
    elbow(apml, 15, "apml", kmeans, 10, euclid, euclidean_centroid, kmeans_pp_init)
    elbow(gauss, 10, "gaussian", kmeans, 10, euclid, euclidean_centroid, kmeans_pp_init)


def silhouette_losses(circles, apml, gauss):
    circles_losses, circles_S_k = silhouette(circles, 10, kmeans, 10, euclid, euclidean_centroid, kmeans_pp_init)
    plot_silhouette(circles_losses[2], circles_S_k, 4, "circles")
    apml_losses, apml_S_k = silhouette(apml, 10, kmeans, 10, euclid, euclidean_centroid, kmeans_pp_init)
    plot_silhouette(apml_losses[7], apml_S_k, 9, "APML")
    gauss_losses, gauss_S_k = silhouette(gauss, 10, kmeans, 10, euclid, euclidean_centroid, kmeans_pp_init)
    plot_silhouette(gauss_losses[2], gauss_S_k, 4, "gaussian")


def eigengap_losses(circles, apml, gauss):
    spectral(circles,
             K=None,
             similarity_param=estimate_sigma(
                 circles, 1, 'circles'),
             similarity=gaussian_kernel)
    spectral(apml,
             K=None,
             similarity_param=estimate_sigma(
                 apml, 1, 'apml'),
             similarity=gaussian_kernel)

    spectral(gauss,
             K=None,
             similarity_param=estimate_sigma(
                 gauss, 2, 'gaussian'),
             similarity=gaussian_kernel)


def plot_clustered(data, clustering, name):
    clustered = data[clustering == 0, :]
    for i in range(1, np.max(clustering) + 1):
        clustered = np.vstack((clustered, data[clustering == i, :]))

    plt.figure()
    plt.imshow(clustered, extent=[0, 1, 0, 1], cmap="hot", vmin=-3, vmax=3)
    plt.colorbar()
    plt.title(name)
    plt.show()


def biological_data_clustering(data):
    elbow(data, 10, "biological data", kmeans, 10, euclid, euclidean_centroid, kmeans_pp_init)
    data_losses, data_S_k = silhouette(data, 10, kmeans, 10, euclid, euclidean_centroid, kmeans_pp_init)
    plot_silhouette(data_losses[2], data_S_k, 4, "biological")

    data_clustering, data_centroids, data_loss, data_losses = kmeans(data,
                                                                         K=4,
                                                                         iterations=10,
                                                                         metric=euclid,
                                                                         center=euclidean_centroid,
                                                                         init=kmeans_pp_init)

    plot_clustered(data, data_clustering, "k-mean++ clustering with k=4 for biological data")

    data_clustering, data_centroids, data_loss, data_losses = spectral(data,
                                                                       K=4,
                                                                       similarity_param=15,
                                                                       similarity=gaussian_kernel)
    plot_clustered(data, data_clustering, "spectral clustering with k=4 for biological data")


def plot_digits(digits):
    n = 10
    img = np.zeros((10 * n, 10 * n))
    for i in range(n):
        ix = 10 * i + 1
        for j in range(n):
            iy = 10 * j + 1
            img[ix:ix + 8, iy:iy + 8] = digits.data[i * n + j].reshape((8, 8))
    plt.figure()
    plt.imshow(img, cmap=plt.cm.binary)
    plt.title('A selection from the 64-dimensional digits dataset')
    plt.show()


def plot_mnist(data, digits, name):
    data_min, data_max = np.min(data, 0), np.max(data, 0)
    data = (data - data_min) / (data_max - data_min)

    plt.figure()
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(digits.target[i]),
                 color=plt.cm.Set1(digits.target[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.title(name)
    plt.show()


def visualize_2D(data, target, name):
    plt.figure()
    plt.scatter(data[:, 0], data[:, 1], c =target, cmap=plt.cm.Spectral)
    plt.title(name)
    plt.show()


def visualize_3D(data, target, name):
    from mpl_toolkits.mplot3d import Axes3D
    ax = Axes3D(plt.figure())
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.scatter(data[:,0], data[:,1], data[:,2], c=target, cmap=plt.cm.Spectral)
    plt.title(name)
    plt.show()


def mnist_clustering():
    digits = datasets.load_digits(n_class=10)
    plot_digits(digits)
    tsne = manifold.TSNE(n_components=2, method='barnes_hut')
    tsne_results = tsne.fit_transform(digits.data)
    plot_mnist(tsne_results, digits, "t-SNE performed on digits")

    pca = decomposition.PCA(n_components=2)
    pca_result = pca.fit_transform(digits.data)
    plot_mnist(pca_result, digits, "PCA performed on digits")

    visualize_3D(manifold.TSNE(n_components=3, method='barnes_hut').fit_transform(digits.data),
                  digits.target,
                  "TSNE in 3D on mnist")

    visualize_3D(decomposition.PCA(n_components=3).fit_transform(digits.data),
                  digits.target,
                  "PCA in 3D on mnist")


def high_dimensional_tsne():
    X, labels = make_blobs(n_samples=700, centers=5, n_features=30,
               cluster_std=4.60, random_state=0)

    visualize_2D(manifold.TSNE(n_components=2, method='barnes_hut').fit_transform(X),
                 labels,
                 "t-SNE performed on high dimensional data")
    visualize_2D(decomposition.PCA(n_components=2).fit_transform(X),
                 labels,
                 "PCA performed on high dimensional data")

    visualize_3D(manifold.TSNE(n_components=3).fit_transform(X),
                 labels,
                 "t-SNE performed on high dimensional data")
    visualize_3D(decomposition.PCA(n_components=3).fit_transform(X),
                 labels,
                 "PCA performed on high dimensional data")


if __name__ == '__main__':

    circles_d = np.array(circles_example().T)
    apml_d = np.array(apml_pic_example())
    gauss_d = np.array(gaussian_example())

    # Question 1: Run K-Means++ on synthetic data
    #############################################
    kmeans_pp_synthetic(circles_d, apml_d, gauss_d)

    # Question 2: Run Spectral Clustering on synthetic data
    #######################################################

    spectral_synthetic_gauss(circles_d, apml_d, gauss_d)
    # Compare weight matrices
    spectral_synthetic_gauss(circles_d, apml_d, gauss_d, True)

    spectral_synthetic_mnn(circles_d, apml_d, gauss_d)
    # Compare weight matrices
    spectral_synthetic_mnn(circles_d, apml_d, gauss_d,True)

    # Question 3: Demonstrate that ou can use the 'elbow' or Silhouette methods to choose k correctly for synthetic data
    ####################################################################################################################
    elbow_losses(circles_d, apml_d, gauss_d)
    silhouette_losses(circles_d, apml_d, gauss_d)
    eigengap_losses(circles_d, apml_d, gauss_d)

    # Question 4: apply K-Means and spectral clustering on biological data.
    #######################################################################

    with open('microarray_data.pickle', 'rb') as f:
        microarray_data = pickle.load(f)
    with open('microarray_genes.pickle', 'rb') as f:
        microarray_genes = pickle.load(f)
    with open('microarray_conds.pickle', 'rb') as f:
        microarray_conds = pickle.load(f)

    biological_data_clustering(microarray_data)


    # Question 5: Familiarize yourself with the t-SNE algorithm using MNIST and a toy dataset.

    mnist_clustering()
    high_dimensional_tsne()