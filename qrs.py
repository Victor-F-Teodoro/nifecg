from generic import Block
from wfdb import processing
from numpy import zeros_like
from scipy import signal
from skimage.restoration import denoise_wavelet
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
	
	def forward(self, sig, ts, flag):
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
			if flag == 1:
				chann = denoise_wavelet(chann, wavelet='db10', mode='soft', wavelet_levels=10	, method='BayesShrink', rescale_sigma='True')
			elif flag == 2:
				chann = signal.savgol_filter(chann, window_length=30, polyorder=4, mode='mirror')
				### FILTRES 2018

				#### FIR FILTER
				# Define the filter specifications
				fs = fs    # Sampling frequency
				f1 = 3      # Lower cut-off frequency
				f2 = 35     # Upper cut-off frequency
				numtaps = 11  # Filter order (number of coefficients)
				nyq = 0.5 * fs

				# Compute the filter coefficients using Hamming window
				taps = signal.firwin(numtaps, [f1/nyq, f2/nyq], pass_zero=False, window='hamming')
				# Filter the signal using the FIR filter
				chann = signal.lfilter(taps, 1.0, chann)

			_, rpeaks = nk.ecg_peaks(chann, sampling_rate=fs, correct_artifacts=True)
			new_sig.append(chann[rpeaks["ECG_R_Peaks"]])
			new_ts.append(ts[rpeaks["ECG_R_Peaks"]])
		
		return new_sig, new_ts