from generic import Block
from wfdb import processing
from numpy import zeros_like
from scipy.signal import find_peaks
import neurokit2 as nk

class PrimaryQrsBlock(Block):

	def __init__(self, threshold=1, adc_gain=2, adc_zero=0):
		"""
		initializes the block

		Parameters:
			threshold : List[float], optional
				threshold for qrs complexes detection for each channel, default=1
			adc_gain : List[float], optional
				adc gain for each channel, default=2
			adc_zero : List[float], optional
				adc zero for each channel, default=0
		Returns:
			None
		"""

		self.th = threshold
		self.adc_gain = adc_gain
		self.adc_zero = adc_zero

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

		fs = 1/(ts[1]-ts[0])
		new_sig = zeros_like(sig)
		n_channels = sig.shape[-1]

		if type(self.th) in [int, float]:
			self.th = [self.th] * n_channels
		if type(self.adc_gain) in [int, float]:
			self.adc_gain = [self.adc_gain] * n_channels 
		if type(self.adc_zero) in [int, float]:
			self.adc_zero = [self.adc_zero] * n_channels 

		i = 0
		for channel, threshold, adc_gain, adc_zero in zip(sig.T, self.th, self.adc_gain, self.adc_zero):
			qrs_locs = processing.gqrs_detect(d_sig=channel, fs=fs, 
				    adc_gain=adc_gain, adc_zero=adc_zero, threshold=threshold)
			if len(qrs_locs) > 0:
				new_sig[qrs_locs,i] = sig[qrs_locs, i]
			i += 1
		return new_sig, ts

class SecondaryQrsBlock(Block):
	
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
		
		fs = 1/(ts[1]-ts[0]) 
		new_sig = []
		new_ts = []
		for chann in sig.T:
			_, rpeaks = nk.ecg_peaks(chann, sampling_rate=fs)
			new_sig.append(chann[rpeaks["ECG_R_Peaks"]])
			new_ts.append(ts[rpeaks["ECG_R_Peaks"]])
		return new_sig, new_ts