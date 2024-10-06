import pandas
from eventdetector_ts.metamodel.meta_model import MetaModel
import numpy as np
import pandas as pd
from obspy import read
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os
from scipy import signal
import hashlib
from matplotlib import cm

base_dir = "./data/lunar/training"
data_dir = "data/S12_GradeA"
catalog_file = base_dir + "/catalogs/apollo12_catalog_GradeA_final.csv"
catalog = pd.read_csv(catalog_file)
events = []
times = np.array([])
velocities = np.array([])
hashes = set()

for (idx, filename) in enumerate(catalog['filename'][0:1]):
    print("----- Reading ", filename, idx, "-----")
    row = catalog.iloc[idx]
    arrival_time = float(row['time_rel(sec)'])
    mseed_file = f'{base_dir}/{data_dir}/{filename}.mseed'

    digest = hashlib.md5(open(mseed_file, 'rb').read()).hexdigest()

    # Add the next mseed values if the data has not already been added
    # Data could already be added if we have two files with the same data
    delta = 0 if len(times) == 0 else times[-1] + times[-1] - times[-2]
    if digest not in hashes:
        # Pad out the previous mseed file
        if len(times) > 1:
            for i in range(0, 100):
                times = np.append(times, times[-1] + times[-1] - times[-2])
                velocities = np.append(velocities, 0)

        st = read(mseed_file)
        delta = 0 if len(times) == 0 else times[-1] + times[-1] - times[-2]
        times = np.append(times, st.traces[0].times() + delta)
        velocities = np.append(velocities, st.traces[0].data)

    events.append(arrival_time + delta)
    events.append(arrival_time + delta + 2)
    hashes.add(digest)

print(events)
print(len(times))
print(len(velocities))

df = pd.DataFrame({"time": times, "velocity": velocities})
df["time"] = pd.to_datetime(df["time"], unit='s')
df.set_index("time", inplace=True)
meta_model = MetaModel(output_dir="model", dataset=df, events=events, width=2, epochs=5)
meta_model.fit()
