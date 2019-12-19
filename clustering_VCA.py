import hyperspy.api as hs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hyperspy.misc.machine_learning import import_sklearn
from VCA import *
from implementations import *
from fcmeans import FCM
from sklearn.cluster import KMeans

#This function returns data after the dimension reduction using SVD
def svd_decomposition( data ):
	U, s, VT = np.linalg.svd(data)
	S = np.diag(s[0:2])
	U1 = U[:,0:2]
	VT1 = VT[0:2,:]
	return U1.dot(S.dot(VT1))

#This function uses the given VCA function to implement vertex component analysis on the
#  given hyperspectral data
def VCA( data ):
	Y = data.transpose()
	Ae, indice, Yp = vca(Y,2)
	plot( Ae.transpose(), 'VCA', 'Decomposition using VCA')

# This function uses the f-cmeans library to perform fuzzy clustering of the data
def fuzzy_clustering( data ):
	svd = svd_decomposition( data )
	fcm = FCM(n_clusters=2, m=1.2, max_iter=250)
	fcm.fit(svd)
	fcm_centers = fcm.centers
	fcm_labels  = fcm.u.argmax(axis=1)
	plot( fcm_centers, 'fuzzy', 'Decomposition using Fuzzy clustering')

# This function uses the k-means given by sklearn to perfrom clustering on the data and
# plot the centers
def K_means( data ):
	svd = svd_decomposition( data )
	kmeans = KMeans(n_clusters=2, random_state=0).fit(svd)
	kmeans_centers = kmeans.cluster_centers_
	plot( kmeans_centers, 'kmeans', 'Decomposition using K-Means clustering')
