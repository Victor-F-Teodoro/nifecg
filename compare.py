from generic import Block
from numpy import correlate, argmax, array

class CompareBlock(Block):
	"""
	Block for comparing the likeness between the channels and the 
	ICA outputs
	"""

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

		primary, secondary, primary_qrs, secondary_qrs = sig

		corr_matrix = array([ [ max(correlate(x,y)) for x in secondary_qrs] for y in primary_qrs ])
		best = argmax(corr_matrix)

		m,n = corr_matrix.shape
		i = best//m
		j = best%n

		return (secondary[:,j], secondary_qrs[j]), ts