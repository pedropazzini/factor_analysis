import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FactorAnalysis,PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sas7bdat import SAS7BDAT
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import pdist,squareform
from scipy.cluster import hierarchy

def fit_factor_analysis(percentage=0.8):
    """
    Runs the factor analysis.

    Parameters:

        percentage: float, default:0.8

        The percentage of the cumulative sum of the eigenvalues to be held. This number defines the number of loading factors in the analysis.

    Returns:
        
        X: array of floats [n_samples,n_factors]

            The transformed data after the factor analysis.

        components: array of floats [n_factors,n_samples]

            The components of the factor analysis
    """
    fa = FactorAnalysis()
    fa.fit(data)
    C = fa.get_covariance()
    l,e = np.linalg.eigh(C)
    cs = np.cumsum(l[::-1])/np.sum(l)
    n = np.sum(cs<percentage)

    fa.n_components = n
    X_ = fa.fit_transform(data)
    components = fa.components_
    return X_,components

def plot(data,names,labels, reduce_by_PCA=True,force_2D=False):
    """
    Plots the data dividing it in clusters and puting a name on each instance.

    Parameters:

        data: array of flaots [n_samples,3]

            Data containing the points to be ploted in 3D or 2D, the 3D version beeing prefered. If the data is not in 3D or 2D it is possibl to reduce it by PCA 
            to be possible the visulazition of the instances. For that the parameter 'reduce_by_PCA' should be setted to true.

        names: array of strings [n_samples,]

            Array containing the names of each instance to be plotted.

        labels: array of integers [n_samples,]

            Label of each instance assinging it to a cluster.

        reduce_by_PCA: boolean, default=False

            Boolean value that indicates if the 'data' parameter is not in 3D or 2D form it should be reduced by PCA to allow visualization in 3D or 2D. 
            If this parameter is not setted and the data is not in 3D or 2D an exception is raised.

        force_2D: boolean, dafault=False

            Boolean value that indicates if the plot should be in 2D.

    Returns:

        void
    """

    if data.shape[1] == 2:
        plot_2D(data,names,labels, reduce_by_PCA)
    else:
        if not force_2D:
            plot_3D(data,names,labels, reduce_by_PCA)
        else:
            plot_2D(data,names,labels, reduce_by_PCA)

def plot_3D(data,names,labels, reduce_by_PCA=True):
    """
    Plots the data dividing it in clusters and puting a name on each instance.

    Parameters:
        
        data: array of flaots [n_samples,3]
        
            Data containing the points to be ploted in 3D. If the data is not in 3D it is possibl to reduce it by PCA to be possible the visulazition of the instances in 3D.
            For that the parameter 'reduce_by_PCA' should be setted to true.

        names: array of strings [n_samples,]

            Array containing the names of each instance to be plotted.

        labels: array of integers [n_samples,]

            Label of each instance assinging it to a cluster.

        reduce_by_PCA: boolean, default=False

            Boolean value that indicates if the 'data' parameter is not in 3D form it should be reduced by PCA to allow visualization in 3D. If this parameter is not setted and 
            the data is not in 3D an exception is raised.

    Returns:
        
        void
    """

    # Check the dimension of data
    if data.shape[1] > 3:
        if not reduce_by_PCA:
            raise ValueError('Data is not in 3D shape[' + str(data.shape[0]) + ',' + str(data.shape[1]) + ']. Set the parameter \'reduce_by_PCA\' to True to allow visualization...')
        else:
            data = reduce_PCA(data,n=3)

    #plot the data
    colors_ = "bgrcmykw"
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x =data[:,0]
    y =data[:,1]
    z =data[:,2]

    for i in range(len(x)):
        ax.scatter(x[i], y[i], z[i], c=colors_[labels[i]], marker='o')
        if len(names) != 0:
            ax.text(x[i],y[i],z[i], names[i] , zorder=1,color='k') 

    plt.show()

def plot_2D(data,names,labels, reduce_by_PCA=True):
    """
    Plots the data dividing it in clusters and puting a name on each instance.

    Parameters:
        
        data: array of flaots [n_samples,2]
        
            Data containing the points to be ploted in 2D. If the data is not in 2D it is possibl to reduce it by PCA to be possible the visulazition of the instances in 2D.
            For that the parameter 'reduce_by_PCA' should be setted to true.

        names: array of strings [n_samples,]

            Array containing the names of each instance to be plotted.

        labels: array of integers [n_samples,]

            Label of each instance assinging it to a cluster.

        reduce_by_PCA: boolean, default=False

            Boolean value that indicates if the 'data' parameter is not in 2D form it should be reduced by PCA to allow visualization in 2D. If this parameter is not setted and 
            the data is not in 2D an exception is raised.

    Returns:
        
        void
    """

    # Check the dimension of data
    if data.shape[1] > 2:
        if not reduce_by_PCA:
            raise ValueError('Data is not in 2D shape[' + str(data.shape[0]) + ',' + str(data.shape[1]) + ']. Set the parameter \'reduce_by_PCA\' to True to allow visualization...')
        else:
            data = reduce_PCA(data,n=2)

    #plot the data
    colors_ = "bgrcmykw"
    unique_labels = np.unique(labels)

    plt.figure()
    
    [plt.plot(i[0],i[1],color=colors_[np.where(unique_labels==labels[pos])[0]],marker='o') for pos,i in enumerate(data)]
    if len(names) != 0:
        [plt.annotate(names[pos], (i[0],i[1])) for pos,i in enumerate(data)]

    plt.show()

def rotate(X):
    """
    Rotates the X matrix

    Paramters:

        X: array of floats [n_samples,n_features]

        The matrix to be rotated.

    Returns:

        X_: array of floats [n_samples,n_features]

        The rotated matrix.
    """

    return X.dot(ortho_rotation(X))

def ortho_rotation(lam, method='varimax',gamma=None,
                           eps=1e-6, itermax=100):
    """
    Return orthogal rotation matrix
    TODO: - other types beyond 
    """
    if gamma == None:
        if (method == 'varimax'):
                gamma = 1.0
        if (method == 'quartimax'):
                gamma = 0.0

    nrow, ncol = lam.shape
    R = np.eye(ncol)
    var = 0

    for i in range(itermax):
        lam_rot = np.dot(lam, R)
        tmp = np.diag(np.sum(lam_rot ** 2, axis=0)) / nrow * gamma
        u, s, v = np.linalg.svd(np.dot(lam.T, lam_rot ** 3 - np.dot(lam_rot, tmp)))
        R = np.dot(u, v)
        var_new = np.sum(s)
        if var_new < var * (1 + eps):
            break
        var = var_new

    return R


def reduce_PCA(data,n=2):
    """
    Reduces the data to the desired dimension by the PCA method.

    Parameters:

        data: array of floats [n_samples,n_features]

            The data to be reduced. If the n_features is less than the 'n' parameter an exception is raised.

        n: integer greather than zero. default=2

            An integer representing representing the number of features of the reduced data. If 'n' is less than n_features an execption is raised.

    Returns:

        new_data: array of floats [n_samples,n]

            The new reduced array.
    """

    # Checks the parameters
    if data.shape[1] <= n:
        raise ValueError('The parameter \'n\' should be greather than \'' + str(data.shape[1]) + '\' to be possible the reduction.')

    pca = PCA(n_components=n)
    new_data = pca.fit_transform(data)

    return new_data

def clusterize_data(data, k=None, range_k=list(range(2,10)),algorithm='k-means'):
    """
    Clusterize the data by the algorithm with the specified value of 'k'. If the parameter 'k' is None it finds the "best" partition within 
    the determinaded range of the number of clusters. The clustering algorithm is iterativily  with all the values of 'k' in the 'range_k' variable 
    and each partition is evaluated by the silhouette index. The one that result in a better index value is returned.

    Parameters:

        data: array of floats [n_samples,n_features]

            The data to be clustered.

        k: integer, greather than 2. default: None

            The number of clusters of the data. If this value is None tham the value is indicated by the silhouette index.

        range_k: list of integers. default [2,...,9]

            The list of the possible number of clusters of data. The lowest value can not be smaller than 2. The greathest value can not be greather than n_samples -1.

        algorithm: string, defaul:'k-means'

            The clustering algorithm to be used. Allowed: ['k-means','hierarchical-average','hierarchical-complete','hierarchical-single']

    Returns:

        labels: list of integers [n_samples,]

            A list of integers assigning each sample to a cluster.
    """

    # Check the input algorithm
    allowed_algs = ['k-means','hierarchical-average','hierarchical-complete','hierarchical-single']
    if algorithm not in allowed_algs:
        raise ValueError('Algorithm not allowed: \'' + algorithm + '\'. Allowed ones: [' + ','.join(allowed_algs) + ']')

    # Check the number of clusters input
    if k is not None:
        if  k < 2:
            raise ValueError('Invalid value of "k". It should be greather than 2')
        else:
            range_k = [k]

    # Set the classifier
    km = None
    Z = None
    if 'k-means' in algorithm:
        km = KMeans()
    else:
        # calculates the matrix distance and obtain the linkage matrix
        D = squareform(pdist(data))
        type_linkage = algorithm.split('-')[1]
        Z = hierarchy.linkage(D,type_linkage)

    labels_k = []
    silhouette_k = []


    # For each value of k clusterize by kmeans and evaluates the silhouette.
    for k in range_k:
        l_k = None
        if 'hierarchical' not in algorithm:
            km.n_clusters = k
            l_k = km.fit_predict(data) 
        else:
            l_k = hierarchy.fcluster(Z,k,criterion='maxclust')

        s_k = silhouette_score(data,l_k)
        labels_k.append(l_k)
        silhouette_k.append(s_k)

    # Finds the labels with the best [maximum] silhouette index and return it.
    return labels_k[np.argmax(silhouette_k)]


def get_data(filename='nmtwins.sas7bdat'):
    """
    Reads the .sas file and returns it as a numpy array. The first column of the data is supposed to be an id. Repeated instances are discarded as well instances with missing data.

    Parameters:

        filename: string, default: nmtwins.sas7bdat

            The name of the file that contains the data.

    Returns:

       header: np.array of strings
            
            The header of the data.

        new_data: np.array of floats

            The data in numpy array format.

    """
    data = []

    with SAS7BDAT(filename) as f:
        for row in f:
            data.append(row)

    data = np.array(data)
    header = data[0]
    data = np.array(data[1:],dtype=np.float64)
    rows,cols = np.where(np.isnan(data))
    urows = np.unique(rows)
    unique_twins_id = np.unique(data[urows,0])
    new_data = np.array([i for i in data if i[0] not in unique_twins_id])

    return header,new_data


if __name__ == "__main__":

    #header, data = get_data('nmtwins.sas7bdat'):
    #header, data = get_data('nmtest_twins.sas7bdat')
    header, data = get_data('wiscsem.sas7bdat')
    labels = data[:,0]
    data = data[:,1:]
    real_header = header[1:]

    transformed_data,components = fit_factor_analysis(percentage=0.8)
    labels_components = clusterize_data(components.T,k=3)
    plot(components.T,real_header,labels_components,force_2D=True)

    components_r = rotate(components.T)
    labels_components_r = clusterize_data(components_r,k=3,algorithm='hierarchical-average')
    plot(components_r,real_header,labels_components_r,force_2D=True)

    labels_transformed = clusterize_data(transformed_data,algorithm='hierarchical-average')
    plot(transformed_data,[],labels_transformed,force_2D=True)

