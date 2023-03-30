import wfdb
import numpy as np
import matplotlib.pyplot as plt
from preprocessing import PreprocessingBlock
from bss import IcaBlock, PcaBlock
from qrs import PrimaryQrsBlock, SecondaryQrsBlock
from compare import CompareBlock

num_samples = 75
sample_names = ["a" + str(i).zfill(2) for i in range(1,num_samples+1)]
scores = []

for sample in sample_names:
    try:
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
            qrs_window = int(300e-3 * fs)  # sampling points inside a 100ms window
            qrs_shift = int(100e-3 * fs)
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
        qrs2_sig, qrs2_ts = block_4.forward(fsig, prep_ts)

        # from Behar 2016, running the results through an ICA block improves quality
        block_5 = IcaBlock()
        ica_sig, ica_ts = block_5.forward(fsig, prep_ts)
        qrs3_sig, qrs3_ts = block_4.forward(ica_sig, ica_ts)

        # copying what Andreotti 2014 did, the best ICA channel will be the one
        # that better matches the fqrs from before the bss
        block_6 = CompareBlock()
        (best_fsig, best_fqrs), best_ts = block_6.forward([fsig, ica_sig, qrs2_ts, qrs3_ts], fts)

        ypred = np.zeros_like(ts)
        ypred[[np.where(ts == t)[0] for t in best_fqrs]] = 1

        ytrue = np.zeros_like(ts)
        ytrue[truth_idxs] = 1

        # assuming a max heart rate of 300bpm
        accepted_decal = int(20e-3 * fs)  # accepted error 

        to_be_masked = np.copy(ypred)  # correct predictions will be removed from this array
        total_positives = np.sum(ytrue)
        true_positives = 0
        false_negatives = 0
        true_negatives = 0
        for idx in truth_idxs:
            if np.any(ypred[idx - accepted_decal : idx + accepted_decal]):
                true_positives += 1
            else:
                false_negatives += 1  
            to_be_masked[idx - accepted_decal : idx + accepted_decal] = 0
        false_positives = np.sum(to_be_masked)

        acc = (true_positives + true_negatives) / (true_positives + false_positives + true_negatives + false_negatives)
        print(f"{sample} : {acc}")
        scores.append(acc*100)
    except Exception:
        scores.append(0)

total_acc = np.mean(scores)

plt.figure(figsize=(12,8), dpi=80)
plt.bar(np.arange(len(scores)) + 1, scores)
plt.ylabel("accuracy (%)")
plt.xlabel("sample in the set-a")
plt.title("algorithm performance")

ax = plt.gca()
ax.text(0.25,0.25, f"overall acc: {total_acc:.2f}", bbox=dict(facecolor='blue', alpha=0.5), horizontalalignment='center',
     verticalalignment='center', transform=ax.transAxes)
plt.show()
