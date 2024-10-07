# Seismic Detection Across the Solar System

One of the main issues surrounding planetary science is the high energy cost of transmitting data back to Earth for processing. One solution to this issue involves processing data on-site, directly on the probes we send to these celestial bodies. Our work over these past two days attempts to do just that. We devised a novel algorithm involving bandpass filters and sliding windows to make accurate predictions about when seismic activity occurred. Our hope is that, with our algorithm, scientists can avoid having to send large amounts of raw data back to Earth.

## Technical Implementation

Our approach involved several distinct phases. The first step was to apply a bandpass filter to all incoming data and only letting through frequencies between 0.5 and 1 Hz. We then created a spectrogram out of this data. Afterwards, we compressed this 2D data into a 1D array by taking the sum of each column. We then determined the maximum subarray of this array. The length of our subarray was determined with a preprocessing step, where we compared the performance of our algorithm with different subarray lengths against the training data. Finally, we converted the start index of the subarray to a timestamp and returned this timestamp as the start of the seismic activity.

## Running our Code

- Create and initialize a virtual environment
- Install the necessary packages from `requirements.txt`
- Run the cells of the `catalog_generation.ipynb` notebook to generate the test data catalogs
- (Optionally) run the cells of the `signal_processing.ipynb` notebook to get a closer look at the window size calculations

## Other Approaches

In creating our algorithm, we initially went down paths that did not yield the results that we had hoped. Rather than simply delete this code, we felt that saving this code was better, so that other people may pick up where we had left off. This code is stored in the `other_approaches` folder.

The `eventdetector` folder contains a starting implementation of using machine learning to predict seismic activity. This was based off of [this paper](https://arxiv.org/pdf/2310.16485) and its associated package. Due to the lack of time and resources, we were unable to train the deep learning model. One area of future exploration would be to finish this work and create the model.

The `cnn` folder contains a starting implementation for training a CNN based off the spectrogram images directly (rather than the algorithmic approach we ended up taking via dimensionality reduction). Like with the `eventdetector` code, time and resource limitations prevented us from fully realizing this model. We encourage others to finish what we had started by training the model.

## Copyright

All code is &copy; Chris Farber and Anish Kambhampati, licensed MIT.
