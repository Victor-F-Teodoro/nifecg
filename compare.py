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
		#corr_matrix = array([ [ max(correlate(x,y)) for x in secondary_qrs] for y in primary_qrs ])
		#best = argmax(corr_matrix)
		k = 0
		max_length = 0
		for l in secondary_qrs:
			##print("len l: ",len(l))
			#print("len comp: ",len(secondary_qrs[k]))
			if len(l)>max_length:
				max_length = len(l)
				o = k
				
			k = k+1
		#m,n = corr_matrix.shape
		#i = best//m
		#j = best%n
		#print("res:" ,len(secondary_qrs[j])-max_length)
		return (secondary[:,o], secondary_qrs[o]), ts