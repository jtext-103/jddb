# This example show how you query shot and read data from if
# import jddb modules

# %%
import matplotlib.pyplot as plt
from jddb.meta_db import MetaDB
from jddb.file_repo import FileRepo
from jddb.processor import Shot
import numpy as np

# %% connect to the MetaDB
connection_str = {
    "host": "localhost",
    "port": 27017,
    "database": "JDDB"
}
collection = "Labels"

db = MetaDB(connection_str, collection)

# %%
#  find all the shot with shot_list in range [10000, 20000] && [IP, BT] tags available && is disruption
shot_list = [shot for shot in range(1000000, 2000000)]
complete_disruption_shots = db.query_valid(
    shot_list=shot_list,
    label_true=["IsDisrupt", "ip", "bt"]
)
print(complete_disruption_shots)
print(len(complete_disruption_shots))
# %%
# find all the shot with IP>200kA, 0.45s<Tcq<0.5s  && is disruption && with those diagnostis [ip, bt] available
ip_range = [160, None]
tcq_range = [0.45, 0.5]
chosen_shots = db.query_range(
    ["IpFlat", "DownTime"],
    lower_limit=[ip_range[0], tcq_range[0]],
    upper_limit=[ip_range[1], tcq_range[1]],
    shot_list=complete_disruption_shots
)
print(chosen_shots)
print(len(chosen_shots))

# %%
# sync meta_db label to shot files in file repo
file_repo = FileRepo(r"..//FileRepo//$shot_2$xx//$shot_1$x//")
file_repo.sync_meta(db)
shot = file_repo.get_all_shots()[0]
print(file_repo.read_labels(shot))

# %%
# plot ip and some diagnostics.
# read one signal with tag
signals = file_repo.read_data(shot, ["\ip"])
data = signals["\ip"]

# read start time and sampling rate to generate x-axis
attribute_list = ["StartTime", "SampleRate"]
attributes = file_repo.read_attributes(shot, "\ip", attribute_list)
start = attributes["StartTime"]
sr = attributes["SampleRate"]
stop = start + len(data) / sr
time_axis = np.linspace(start, stop, num=len(data), endpoint=True)

# plot
# plt.figure()
# plt.plot(time_axis, data)
# plt.show()

# an alternative way to read a shot
example_shot = Shot(shot, file_repo)
# get a signal instance from a shot instance
ip = example_shot.get('\\ip')
# an alternative way to plot with signal
plt.plot(ip.time, ip.data)
plt.show()
# %%
# disconnect from MongoDB
db.disconnect()
