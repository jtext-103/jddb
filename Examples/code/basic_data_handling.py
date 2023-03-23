# This example show how you query shot and read data from if
# import jddb modules
from JDDB import meta_db
from JDDB import file_repo

# connect to the meta_db
connection_str = {
            "host" : "localhost",
            "port" : 27017,
            "username" : "DDBUser",
            "password" : "*******",
            "database": "JDDB"
          }
collection = "Labels"
c = meta_db.ConnectDB()
labels = c.connect(connection_str, collection)

# connnect to the FileDB
FileDB = meta_db.MetaDB(labels)

# find all the shot with shot_no in range [10000, 20000] && [IP, BT] available && is disruption
shot_list = [shot for shot in range(10000, 20000 + 1)]
complete_disruption_shots = FileDB.query_valid(
                shot_list=shot_list,
                label_true=["IsDisrupt", "ip", "bt"]
)

# find all the shot with IP>200kA, Tcq<0.6s  && is disruption && with those diagnostis [ip, bt] available
ip_range = [200, None]
tcq_range = [None, 0.6]
chosen_shots = FileDB.query_range(
                ["IpFlat", "DownTime"],
                lower_limit=[ip_range[0], tcq_range[0]],
                upper_limit=[ip_range[1], tcq_range[1]],
                shot_list=complete_disruption_shots
)

# sync meta_db label to shot file, shot_no [10000, 10001]
file_repo.sync_meta(FileDB, [10000, 10001])

# plot ip and some diagnostics.
shot_no = 10000
data_dict = file_repo.read_data(shot_no, ["\ip"])
data = data_dict["\ip"]
attribute_list = ["StartTime", "SampleRate"]
attribute_dict = file_repo.read_attributes(shot_no, "\ip", attribute_list)
start = attribute_dict["StartTime"]
fs = attribute_dict["SampleRate"]
stop = start + len(data) / fs
time_axis = np.linspace(start, stop, num=len(data), endpoint=True)

import matplotlib.pyplot as plt
plt.figure()
plt.plot(time_axis, data)
plt.show()

# disconnect from MongoDB
c.disconnect()



