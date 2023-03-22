import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class Block(ABC):

	@abstractmethod
	def forward(self, sig, ts):
		"""
		abstract method for treating the signal, must be overwritten

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
		raise NotImplementedError
