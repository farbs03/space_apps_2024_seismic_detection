from obspy.signal.invsim import cosine_taper
from obspy.signal.filter import highpass
from obspy.signal.trigger import classic_sta_lta, plot_trigger, trigger_onset
from obspy import read
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from matplotlib import cm
from datetime import datetime
import hashlib

base_dir = "./data/lunar/training"
data_dir = "data/S12_GradeA"
catalog_file = base_dir + "/catalogs/apollo12_catalog_GradeA_final.csv"
catalog = pd.read_csv(catalog_file)

hashes = set()


def max_subarray_sum_of_length_k(arr, k):
    # Check if array has enough elements
    if len(arr) < k:
        return None, None

    # Compute the sum of the first subarray of length k
    max_sum = current_sum = sum(arr[:k])
    max_start_index = 0

    # Slide the window over the array
    for i in range(k, len(arr)):
        # Update the current sum by sliding the window one element forward
        current_sum += arr[i] - arr[i - k]

        # If the new sum is larger, update max_sum and max_start_index
        if current_sum > max_sum:
            max_sum = current_sum
            max_start_index = i - k + 1

    # Return the starting index and the maximum sum
    return max_start_index, max_sum


def pad_spectrogram(s, target_shape):
    current_height, current_width = s.shape
    target_height, target_width = target_shape

    # Calculate how much padding is needed for height and width
    pad_height = target_height - current_height
    pad_width = target_width - current_width

    # Apply padding if needed
    if pad_height > 0 or pad_width > 0:
        s_padded = np.pad(s,
                          ((0, pad_height), (0, pad_width)),  # Pad along height and width
                          mode='constant', constant_values=0)
    else:
        s_padded = s  # If already the right shape

    return s_padded

difference = np.array([])

for (idx, filename) in enumerate(catalog['filename']):
    print("----- Reading ", filename, idx, "-----")
    row = catalog.iloc[idx]
    arrival_time = float(row['time_rel(sec)'])
    mseed_file = f'{base_dir}/{data_dir}/{filename}.mseed'

    st = read(mseed_file)
    tr = st.traces[0].copy()
    tr_times = tr.times()
    tr_data = tr.data

    digest = hashlib.md5(open(mseed_file, 'rb').read()).hexdigest()
    if digest in hashes:
        print("DUPLICATE FILE")
        #continue
    hashes.add(digest)

    # Set the minimum frequency
    minfreq = 0.1
    maxfreq = 3

    # Going to create a separate trace for the filter data
    st_filt = st.copy()
    st_filt.filter('bandpass', freqmin=minfreq, freqmax=maxfreq)
    tr_filt = st_filt.traces[0].copy()
    tr_times_filt = tr_filt.times()
    tr_data_filt = tr_filt.data

    # Sampling frequency of our trace
    df = tr_filt.stats.sampling_rate

    f, t, sxx = signal.spectrogram(tr_data_filt, tr_filt.stats.sampling_rate)
    # Start time of trace (another way to get the relative arrival time using datetime)
    arrival_time = datetime.strptime(row['time_abs(%Y-%m-%dT%H:%M:%S.%f)'], '%Y-%m-%dT%H:%M:%S.%f')
    starttime = tr.stats.starttime.datetime
    arrival = (arrival_time - starttime).total_seconds()


    #threshold = 1e-20  # Adjust this based on your data
    #sxx_filtered = np.where(sxx < threshold, 0, sxx)
    #sxx_padded = pad_spectrogram(sxx_filtered, (129, 2555))
    #slog = np.log10(sxx_padded)
    slog = sxx.copy()

    sums = np.array([])
    for i in range(0, len(slog[0])):
        sums = np.append(sums, 0)
        for j in range(0, len(slog)):
            sums[-1] += slog[j][i]

    start_index, max_sum = max_subarray_sum_of_length_k(sums, 50)
    predicted = (tr_times_filt[-1] - tr_times_filt[0]) / slog.shape[1] * start_index
    print(f"Start index: {start_index}")
    print(f"Max sum: {max_sum}")
    print(f"Actual: {arrival}")
    print(f"Predicted: {predicted}")
    difference = np.append(difference, abs(arrival - predicted))

    # Plot the time series and spectrogram
    '''fig = plt.figure(figsize=(10, 10))
    ax = plt.subplot(2, 1, 1)
    # Plot trace
    ax.plot(tr_times_filt, tr_data_filt)

    # Mark detection
    ax.axvline(x=arrival, color='red', label='Detection')
    ax.legend(loc='upper left')

    # Make the plot pretty
    ax.set_xlim([min(tr_times_filt), max(tr_times_filt)])
    ax.set_ylabel('Velocity (m/s)')
    ax.set_xlabel('Time (s)')

    ax2 = plt.subplot(2, 1, 2)
    vals = ax2.pcolormesh(t, f, sxx_padded, cmap=cm.jet, vmax=5e-17)
    ax2.set_xlim([min(tr_times_filt), max(tr_times_filt)])
    ax2.set_xlabel(f'Time (Day Hour:Minute)', fontweight='bold')
    ax2.set_ylabel('Frequency (Hz)', fontweight='bold')
    ax2.axvline(x=arrival, c='red')
    cbar = plt.colorbar(vals, orientation='horizontal')
    cbar.set_label('Power ((m/s)^2/sqrt(Hz))', fontweight='bold')
    print()'''
    #plt.show()

#print(f"Average difference: {difference / len(catalog['filename'])}")
print(f"Average difference: {np.mean(difference), np.median(difference)}")
