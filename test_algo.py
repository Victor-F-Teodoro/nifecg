import wfdb
import numpy as np
from preprocessing import PreprocessingBlock
from bss import IcaBlock, PcaBlock
from qrs import PrimaryQrsBlock, SecondaryQrsBlock
from compare import CompareBlock

num_samples = 75
sample_names = ["a" + str(i).zfill(2) for i in range(1,num_samples+1)]

for sample in sample_names:
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

    temp = np.hstack(zs)
    z = np.zeros_like(temp)
    _, M4 = z.shape
    M = M4//4
    z[:, ::4] = temp[:, :M]
    z[:, 1::4] = temp[:, M:2*M]
    z[:, 2::4] = temp[:, 2*M:3*M]
    z[:, 3::4] = temp[:, 3*M:4*M]

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