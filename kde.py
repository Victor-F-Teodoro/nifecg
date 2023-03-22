from scipy import stats
from numpy import flatten, linspace
from generic import Block

class KdeBlock(Block):
	"""
	gaussian kernel density estimation of the peaks distributuins
	in multiple channels 
	"""

	def forward(self, sig, ts):
		X = flatten(sig)
		kernel = stats.gaussian_kde(X)
		position =  linspace(min(X), max(X), 10000)
		estimated = kernel(position)
		return estimated, position
