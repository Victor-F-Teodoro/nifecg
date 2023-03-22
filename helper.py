from generic import Block
from numpy import stack

class StackBlock(Block):

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

		if not hasattr(self, "sig"):
			self.sig = sig
			return sig, ts

		self.sig = stack((self.sig, sig), axis=-1)
		return self.sig, ts