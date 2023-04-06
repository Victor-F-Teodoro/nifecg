import wfdb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from scipy import signal
from skimage.restoration import denoise_wavelet
import neurokit2 as nk
from preprocessing import PreprocessingBlock
from bss import IcaBlock, PcaBlock
from qrs import PrimaryQrsBlock, SecondaryQrsBlock
from compare import CompareBlock
from scipy.signal import kaiserord, lfilter, firwin, freqz
res_a = []
#for z in range(10,40):
    
sample = "a01"
record = wfdb.rdrecord(f"samples/{sample}")
truth_idxs = wfdb.rdann(f"samples/{sample}","fqrs").sample

signals, fs, adc_data = record.p_signal, record.fs, (record.adc_gain[0], record.adc_zero[0])
unit = record.units

# constructing time samples
ts = np.linspace(0, len(signals)/fs, len(signals))
truth_ts = ts[truth_idxs]

# preprocessing
block_0 = PreprocessingBlock()
prep_sig, prep_ts = block_0.forward(signals, ts)


# mother (primary) qrs detection (its used gqrs for this task)
block_2 = PrimaryQrsBlock(100, *adc_data)
qrs1_sig, qrs1_ts = block_2.forward(prep_sig, prep_ts)
 
np.savetxt('sig_mae.txt', prep_sig, delimiter=',')
np.savetxt('qrs_mae.txt', qrs1_sig, delimiter=',')
# plottingintermediate results
plt.figure(1)
i = 1
for qrs, channel in zip(qrs1_sig.T, prep_sig.T):
    qrs_indexes = np.where(qrs)
    qrs_ts = prep_ts[qrs_indexes]    

    plt.subplot(prep_sig.shape[-1],1,i)
    plt.plot(qrs_ts, qrs[qrs_indexes], 'rx')
    plt.plot(prep_ts, channel)    
    plt.xlim((38,42))
    i+=1

# function for creating the qrs-complexes extracts
def create_epochs(start, window, sig):
    # setting up where the epoch starts and ends
    epoch_indexes = np.repeat(start,2)
    epoch_indexes[1::2] += window 
    epoch_indexes = epoch_indexes[epoch_indexes < sig.size]

    # splitting at the specified places and selecting the desired parts
    return np.vstack(np.split(sig, epoch_indexes)[1:-2:2]).T, epoch_indexes

# creating the matrix for the PCR #
zs = []
stored_qrs_epochs = []
stored_epoch_indexes = []

for thissig in prep_sig.T:
    ### qrs epoch decomposition ###
    qrs_indexes = np.where(qrs1_sig[:,0] != 0)[0]
    qrs_window = int(150e-3 * fs)  # sampling points inside a 100ms window
    qrs_shift = int(50e-3 * fs)
    qrs_epochs, epoch_indexes = create_epochs(qrs_indexes-qrs_shift, qrs_window, thissig)
    zs.append(qrs_epochs)
    stored_qrs_epochs.append(qrs_epochs)
    stored_epoch_indexes.append(epoch_indexes)

z = np.hstack(zs)

### PCR ###
block_3 = PcaBlock()
pcr, _ = block_3.forward(z.T, None)

# Removing mECG from the signals #
fsig = np.copy(prep_sig)
fts = np.copy(prep_ts)
i = 0
for qrs_epochs, epoch_indexes in zip(stored_qrs_epochs, stored_epoch_indexes):
    for s,e in zip(epoch_indexes[::2], epoch_indexes[1::2]):
        idx = np.arange(s,e)
        qrs_complex = fsig[idx,i]
        fsig[idx,i] = qrs_complex - pcr.predict(qrs_complex.reshape(1,-1))
    i += 1

# fetal (secondary) qrs detection (it's used neurokits' qrs detection algorithm)
block_4 = SecondaryQrsBlock()
qrs2_sig, qrs2_ts = block_4.forward(fsig, prep_ts, 1)

plt.figure(2)
plt.subplot(3,1,1)
plt.plot(prep_ts, prep_sig[:,0])
plt.ylabel("amplitude ({unit[0]})")
plt.xlabel("time (s)")
plt.xlim((38,42))
plt.title("Original Channel")

plt.subplot(3,1,2)
plt.plot(fts, fsig[:,0])
plt.plot(qrs2_ts[0], qrs2_sig[0], "rx")
plt.ylabel(f"amplitude ({unit[0]})")
plt.xlabel("time (s)")
plt.xlim((38,42))
plt.title("Removed mECG")

plt.subplot(3,1,3)
plt.plot(prep_ts, prep_sig[:,0])
plt.plot(fts, fsig[:,0])
plt.ylabel(f"amplitude ({unit[0]})")
plt.xlabel("time (s)")
plt.xlim((38,42))
plt.title("Overlap")
plt.tight_layout()

# from Behar 2016, running the results through an ICA block improves quality
block_5 = IcaBlock()
ica_sig, ica_ts = block_5.forward(fsig, prep_ts)
qrs3_sig, qrs3_ts = block_4.forward(ica_sig, ica_ts, 4)

# plotting all fetal channels
plt.figure(3)
fig = plt.gcf()
fig.suptitle("fECG channels")
i = 1
for hr_sig, hr_ts, sig in zip(qrs2_sig, qrs2_ts, fsig.T):
    plt.subplot(fsig.shape[-1],1,i)
    plt.plot(fts, sig)
    #plt.plot(hr_ts, hr_sig, "rx")
    plt.ylabel(f"amplitude ({unit[0]})")
    plt.xlabel("time (s)")
    plt.xlim((38,42))
    i += 1



# plotting all ICA channels
plt.figure(4)
fig = plt.gcf()
fig.suptitle("ICA channels")
i = 1
for hr_sig, hr_ts, sig in zip(qrs3_sig, qrs3_ts, ica_sig.T):
    plt.subplot(ica_sig.shape[-1],1,i)
    plt.plot(ica_ts, sig)
    plt.plot(hr_ts, hr_sig, "rx")
    plt.ylabel(f"amplitude ({unit[0]})")
    plt.xlabel("time (s)")
    plt.xlim((38,42))
    i += 1

# copying what Andreotti 2014 did, the best ICA channel will be the one
# that better matches the fqrs from before the bss
block_6 = CompareBlock()
(best_fsig, best_fqrs), best_ts = block_6.forward([fsig, ica_sig, qrs2_ts, qrs3_ts], fts)

plt.figure(5)
plt.plot(best_ts, best_fsig)
plt.vlines(truth_ts, ymin=0, ymax=max(best_fsig)*1.3, colors="green", linestyles="dashed")
plt.plot(best_fqrs, max(best_fsig)*np.ones_like(best_fqrs), "rx")
#plt.plot(rpeaks_1['ECG_R_Peaks']/1000, max(best_fsig)/2*np.ones_like(rpeaks_1['ECG_R_Peaks']), "rx")
#plt.plot(rpeaks_2['ECG_R_Peaks']/1000, max(best_fsig)/3*np.ones_like(rpeaks_2['ECG_R_Peaks']), "rx")
plt.title("Extracted Fetal Heart Rate")
plt.ylabel(f"amplitude")
plt.xlabel("time (s)")
plt.xlim((38,42))
plt.legend(["fECG","true fQRS","estimated fQRS",])

plt.show()

