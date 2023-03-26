from sklearn.decomposition import FastICA, PCA
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import RobustScaler

from generic import Block

class IcaBlock(Block):

	def __init__(self):
		self.scale = None

	def forward(self, sig, ts):
		"""
		method for treating the signal

		Parameters:
			sig : ndarray
				signal data
			ts : ndarray
				signal timestamps
		Returns:
			sig : ndarray
				treated signal
			ts : ndarray
				treated signal timestamps
		"""

		# FIXME # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
		# should be using tanh as the g function, can't figure out why it's not working					  # 
		# transformer = FastICA(whiten="arbitrary-variance", fun=lambda x: (np.tanh(x), 1/np.cosh(x)**2)) # 
		# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
		transformer = FastICA(whiten="arbitrary-variance") 
		X_transformed = transformer.fit_transform(sig)
		
		return X_transformed, ts

class PcaBlock(Block):
	standarize = False

	def forward(self, sig, ts):
		"""
		method for treating the signal

		Parameters:
			sig : ndarray
				signal data
			ts : ndarray
				signal timestamps
		Returns:
			sig : ndarray
				treated signal
			ts : ndarray
				treated signal timestamps
		"""

		transformer = make_pipeline(PCA(n_components=10), LinearRegression())
		if self.standarize:
			transformer = make_pipeline(RobustScaler(), PCA(n_components=10), LinearRegression())
		
		X_transformed = transformer.fit(sig,sig)
		
		return X_transformed, ts