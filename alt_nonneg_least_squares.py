import numpy as np

# Alternating least squares. It is used to intialize the parameters of Alternate Non-negative least squares.
def ALS(V,W,H,max_iters,bkg):
	for i in range(max_iters):
		W = np.transpose( fit_neg(np.transpose(V),np.transpose(H),np.transpose(W)) )
		H = fit_neg(V,W,H)
		# As Heuristic is appended to first row we fix the enteries in our intialization.
		# This is one of the ways we use the heuristic.
		H[0,:] = bkg
	W[ np.where(W < 0) ] = 0  
	H[ np.where(H < 0) ] = 0  
	return W,H

# This solves the normal linear regression used in above ALS in each alternating step.
def fit_neg(V,W,H):
	rows = V.shape[0]
	columns = V.shape[1]
	rows_H = H.shape[0]
	for column in range(columns):
		y = V[:,column].reshape((rows,1))
		y = (np.linalg.inv((W.transpose()@W))@W.transpose())@y
		H[:,column] = y.reshape( rows_H )
	return H

# Alternating Non-negative least squares.
# Alternatly fits each matrix and uses the parameters lambda_Si, lambda_bkg to converge to relevant data.
def ANLS(V_,W_,H_,max_iters,lambda_Si,lambda_bkg,gamma,Si_length):
	# Copy the values so the intial values are not overwritten.
	V = np.copy(V_)
	W = np.copy(W_)
	H = np.copy(H_)
	for i in range(max_iters):
		W = np.transpose( fit_W(np.transpose(V),np.transpose(H),np.transpose(W),lambda_bkg,gamma) )
		H = fit_H(V,W,H,lambda_Si,gamma,Si_length)
	return W,H

def fit_H(V,W,H,lambda_Si,gamma,Si_length):
	columns = V.shape[1]
	Si_range = [220-Si_length,220+Si_length]
	Si = H[1,Si_range[0]:Si_range[1]]
	max_iters = 10		 
	avg = 0
	for column in range(columns):
		# If the columns are in the Silicon range then calculate the gradient
		# The gradient is sum of the differences of consecutives sqaures in the Si_range specified 
		reg_column = (column <= Si_range[1]) & (column >= Si_range[0])
		# Here we pass what value we need to subtract finally after the calculated gradient.
		if(column == Si_range[0]):
			# One of the corner.
			avg = 2*H[1,column+1]
		elif (column == Si_range[1]):
			# The Other corner.
			avg = 2*H[1,column-1]
		elif reg_column:
			# For any intermediate value the gradient if the regualrized loss will be ( 2*W - ( W_left + W_right ) ).
			# Calculate  W_left + W_right here and send as a argument.
			avg = H[1,column-1] + H[1,column+1]
		H[:,column] = non_neg_grad_descent_H(V[:,column],W,H[:,column],max_iters,gamma,reg_column,lambda_Si,avg)
	return H

def fit_W(V,W,H,lambda_bkg,gamma):
	columns = V.shape[1]
	max_iters = 10
	for column in range(columns):
		# If it is the first column the regularize its second entry.
		H[:,column] = non_neg_grad_descent_W(V[:,column],W,H[:,column],max_iters,gamma,(column == 0),lambda_bkg)
	return H

# Fit each row of the W.
def non_neg_grad_descent_W(y, X, W, max_iters,gamma, reg_column, lambda_bkg):
	for i in range(max_iters):
		gradient = compute_gradient( y, X, W )
		if( reg_column ):
			# If the column need to be regularized then subtract the gradient appropriately.
			# Note this tries to reduce the value of element W[0,1] in the original decomposition.
			W[1] = W[1] - gamma * lambda_bkg * W[1]
		W = W - gamma * gradient
		W[ np.where(W < 0) ] = 0  
	return W

# Fit each column of the H
def non_neg_grad_descent_H(y, X, W, max_iters,gamma, reg_column, lambda_Si, avg):
	for i in range(max_iters):
		gradient = compute_gradient( y, X, W ) 
		if( reg_column ):
			# If the column need to be regularized then subtract the gradient appropriately.
			W[1] = W[1] - gamma * lambda_Si * ( 2*W[1] - avg ) -  gamma * lambda_Si * W[1]
		W = W - gamma * gradient
		W[ np.where(W < 0) ] = 0  
	return W

# Compute the Gradient wrt w in the loss function (y - Xw)^2
def compute_gradient(y, X, w ):
    error = y - np.dot( X, w )
    dim = y.shape[0]
    gradient = np.dot( np.transpose( X ) , error )*( 1.0/dim )*-1 
    return gradient

# Print the loss of the decomposition 
def print_loss( V, W, H ):
	diff = V - np.matmul( W,H )
	diff = np.linalg.norm( diff )
	print( "Loss" , diff  )









