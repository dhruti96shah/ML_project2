import numpy as np
import pandas as pd
import cv2
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

# Get the heuristic of component B by getting the mask and applying on the data to integrate.
def get_Component_B( data, loadings ):
	# Generate the mask.
	back = loadings[0,:,:]
	ele = loadings[1,:,:]
	nav_mask = (np.divide(back,(ele+back)))*255
	img = np.array(nav_mask, dtype=np.uint8)

	# Use connected components to fix the outliers.
	ret, binary_map = cv2.threshold(img,150,255,0)
	nlabels, labels, stats, centroids  = cv2.connectedComponentsWithStats(np.invert(binary_map), None, None, None, 8, cv2.CV_32S)
	areas = stats[1:,cv2.CC_STAT_AREA]
	result = np.zeros((labels.shape), np.uint8)
	for i in range(0, nlabels - 1):
	    if areas[i] >= 200:   #keep
	        result[labels == i + 1] = 255

	# Smooth the mask using a gaussian filter
	result1 = gaussian_filter(result, sigma=2)>100

	# Integrate on the pure regions on the mask to generate the Heuristic.
	mean = np.zeros(data.shape[2])
	count = 0
	for i in range(100):
	    for j in range(100):
	        if( result1[i,j] == 0 ):
	            mean = mean + data[i,j]*(1/np.linalg.norm(data[i,j]))
	            count = count + 1
	bkg = mean / count
	bkg = bkg.reshape(1,data.shape[2])
	return bkg

# Reshape the data to use in the NMF 
def reshape(data, loadings, factors, bkg ):
	loadings = loadings.reshape( 2, 100*100 ).transpose() 
	data = data.reshape(100*100,2048)
	# truncate the data to few value for computational purposes
	data = data[:,:800]
	factors =  factors[:,:800]
	bkg = bkg[:,:800]
	return data, loadings, factors, bkg

# Append our Heuristic as first row of the data and make factors compatible with it.
def append_Component_B( data, loadings, factors, bkg ):
	data,loadings = normalize( data, loadings, factors )
	# normalize bkg before appending as data is already normalized.
	bkg = bkg * ( 1/np.linalg.norm(bkg) )
	data = np.concatenate( (bkg,data), axis = 0 )

	# Append a row [1,0] to loadings corresponding to bkg
	row_1_0 =  np.array([1,0]).reshape(1,2)
	loadings = np.concatenate(( row_1_0, loadings ), axis = 0)

	# Set the heuristic as the factors for bkg.
	factors[0,:] = bkg

	return data, loadings, factors

# Normalize our data and factors and keeping the invarinant data = loadings * factors.
def normalize( data, loadings, factors ):
	norm_row = np.linalg.norm(data, axis=1)
	data = data / norm_row[:,None]
	loadings = loadings / norm_row[:,None]
	normF = np.zeros( (2, 2) )
	normF[0,0] = 1/np.linalg.norm(factors[0,:])
	normF[1,1] = 1/np.linalg.norm(factors[1,:])
	factors = np.matmul( normF, factors )
	loadings = np.matmul( loadings, np.linalg.inv(normF) )
	return data,loadings

# To run the experiments for various values of hyper parameters and pick the relevant result.
def tune_hyper_parameters(data,loadings,factors):
	gamma_range = [0.001,0.01,0.1,1]
	lambda_range = [0.001,0.01,0.1]
	best_loss = 10e6
	best_gamma = 0
	best_lambda = 0
	for gamma in gamma_range:
	    for lambda_ in lambda_range: 
	        W,H = alt_least_squares(data,loadings,factors,20,lambda_,lambda_,gamma,10)
	        current_loss = print_loss(temp,W,H)
	        if current_loss<= best_loss:
	            best_loss = current_loss 
	            best_gamma = gamma
	            best_lambda= lambda_
	return best_lambda,best_gamma

# Plot the normalized components with appropriate title and save them.
def plot(H,filename,title):
	plt.plot(H[1,:250]/np.linalg.norm(H[1,:]), label='Phase A')
	plt.plot(H[0,:250]/np.linalg.norm(H[0,:]), label='Phase B')
	phase_a = H[1,:]/np.linalg.norm(H[1,:])
	Si = np.mean( phase_a[210:230] )
	title = title + " silicon score " + str( Si )
	plt.legend()
	plt.xlabel('Energy level')
	plt.ylabel('Intensity')
	plt.title(title)
	plt.savefig(filename)
	plt.close()

# 20,10,10,1,10 - best till now
