from obspy.signal.invsim import cosine_taper
from obspy.signal.filter import highpass
from obspy.signal.trigger import classic_sta_lta, plot_trigger, trigger_onset
from obspy import read
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

base_dir = "./data/lunar/training"
data_dir = "data/S12_GradeA"
catalog_file = base_dir + "/catalogs/apollo12_catalog_GradeA_final.csv"
catalog = pd.read_csv(catalog_file)

for (idx, filename) in enumerate(catalog['filename'][0:2]):
    print("----- Reading ", filename, idx, "-----")
    row = catalog.iloc[idx]
    arrival_time = float(row['time_rel(sec)'])
    mseed_file = f'{base_dir}/{data_dir}/{filename}.mseed'

    st = read(mseed_file)
    tr = st.traces[0].copy()
    tr_times = tr.times()
    tr_data = tr.data

    # Set the minimum frequency
    minfreq = 0.1
    maxfreq = 1

    # Going to create a separate trace for the filter data
    st_filt = st.copy()
    st_filt.filter('bandpass', freqmin=minfreq, freqmax=maxfreq)
    tr_filt = st_filt.traces[0].copy()
    tr_times_filt = tr_filt.times()
    tr_data_filt = tr_filt.data

    # Sampling frequency of our trace
    df = tr_filt.stats.sampling_rate

    # How long should the short-term and long-term window be, in seconds?
    sta_len = 240
    lta_len = 600

    # Run Obspy's STA/LTA to obtain a characteristic function
    # This function basically calculates the ratio of amplitude between the short-term
    # and long-term windows, moving consecutively in time across the data
    cft = classic_sta_lta(tr_data_filt, int(sta_len * df), int(lta_len * df))

    # Plot characteristic function
    fig,ax = plt.subplots(1,1,figsize=(12, 3))
    ax.plot(tr_times_filt, cft)
    ax.set_xlim([min(tr_times_filt),max(tr_times_filt)])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Characteristic function')

    # Play around with the on and off triggers, based on values in the characteristic function
    thr_on = 5
    thr_off = 1.5
    on_off = np.array(trigger_onset(cft, thr_on, thr_off))
    # The first column contains the indices where the trigger is turned "on".
    # The second column contains the indices where the trigger is turned "off".

    # Plot on and off triggers
    fig, ax = plt.subplots(1, 1, figsize=(12, 3))
    for i in np.arange(0, len(on_off)):
        triggers = on_off[i]
        ax.axvline(x=tr_times_filt[triggers[0]], color='red', label='Trig. On')
        ax.axvline(x=tr_times_filt[triggers[1]], color='purple', label='Trig. Off')

    # Plot seismogram
    ax.plot(tr_times_filt, tr_data_filt)
    ax.set_xlim([min(tr_times_filt), max(tr_times_filt)])
    ax.legend()
    plt.show()