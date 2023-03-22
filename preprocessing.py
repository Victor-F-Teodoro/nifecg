from scipy import signal
from generic import Block
import numpy as np

class PreprocessingBlock(Block):
	
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

		# assuming the sampling frequency is constant
		fs = 1/(ts[1] - ts[0])

		# filtering frequencies
		wl = 8   # Band Pass
		wh = 80  # Band Pass
		ws1 = 50 # notch-filter
		ws2 = 60 # notch-filter

		# filter definitions
		b,a = signal.butter(4, [wl, wh], "bandpass", fs=fs)
		bs1, as1 = signal.butter(4, [ws1-2, ws1+2], "stop", fs=fs)
		bs2, as2 = signal.butter(4, [ws2-2, ws2+2], "stop", fs=fs)

		# filtering
		sig = np.nan_to_num(sig,nan=0)
		sig = signal.filtfilt(b,a,sig, axis=0)
		sig = signal.filtfilt(bs1,as1,sig, axis=0)
		sig = signal.filtfilt(bs2,as2,sig, axis=0)
		
		return sig, ts
