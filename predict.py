from eventdetector_ts.prediction.prediction import predict
from eventdetector_ts.prediction.utils import plot_prediction
import pandas as pd
import hashlib

data = pd.read_csv('./data/lunar/training/data/S12_GradeA/xa.s12.00.mhz.1970-01-19HR00_evid00002.csv')
inputs = data[["time_rel(sec)", "velocity(m/s)"]]
inputs["time_rel(sec)"] = pd.to_datetime(inputs["time_rel(sec)"], unit='s')
inputs.set_index("time_rel(sec)", inplace=True)
print(inputs)


predicted_events, predicted_op, filtered_predicted_op = predict(dataset=inputs,
                                                                path='model')
print(len(predicted_events))
plot_prediction(predicted_op=predicted_op, filtered_predicted_op=filtered_predicted_op)
