import warnings
import numpy as np
from MDSplus import connection
from typing import List

from .file_repo import FileRepo
import time as delay_time


class MDSDumper:
    def __init__(self, host_name: str, tree_name: str):
        self.host_name = host_name
        self.tree_name = tree_name
        self.conn = None

    def connect(self):
        self.conn = connection.Connection(self.host_name)
        return self.conn

    def disconnect(self):
        if self.conn is not None:
            self.conn.disconnect()

    def dumper(self, file_repo: FileRepo, shot_list: List[int], tag_list: List[str], overwrite=False):
        i = int(0)
        while True:
            if i > len(shot_list) - 1:
                break
            shot = int(shot_list[i])
            try:
                self.conn = self.connect()
            except:
                warnings.warn("Connect Error", category=UserWarning)
                delay_time.sleep(5)
                warnings.warn("Delay 5s and reconnect", category=UserWarning)
                continue
            i += 1
            try:
                self.conn.openTree(self.tree_name, shot=shot)
                for tag in tag_list:
                    data_dict = dict()
                    try:
                        data_raw = np.array(self.conn.get(tag))
                        time_raw = np.array(self.conn.get(r'DIM_OF(BUILD_PATH({})'.format(tag)))
                        fs = len(time_raw) / (time_raw[-1] - time_raw[0]) if len(time_raw) > 1 else 0
                        st = time_raw[0] if len(time_raw) > 1 else 0
                        data_dict[tag] = data_raw
                        attribute_dict = dict()
                        attribute_dict["SampleRate"] = fs
                        attribute_dict["StartTime"] = st
                        file_repo.write_attributes(shot, tag, attribute_dict, overwrite)
                        del attribute_dict
                    except ValueError("Could not read data from {}".format(tag)):
                        continue
                    file_repo.write_data(shot, data_dict, overwrite)
                    del data_dict
                self.conn.disconnect()
            except ConnectionError("Could not open the tree of shot {}".format(shot)):
                raise ConnectionError
