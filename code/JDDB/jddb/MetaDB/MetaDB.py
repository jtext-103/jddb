import os
import numpy as np
from pymongo import MongoClient


class ConnectDB(object):
    def __init__(self):
        self.client = None

    def connect(self, connection_str, collection):
        self.client = MongoClient(connection_str["host"], int(connection_str["port"]))
        db = self.client[connection_str["database"]]
        if ("username" in connection_str.keys()) & ("password" in connection_str.keys()):
            db.authenticate(connection_str["username"], connection_str["password"])
        labels = db[collection]
        return labels

    def disconnect(self):
        if self.client is not None:
            self.client.close()


class UpdateDB(object):
    def __init__(self, label_collection):
        self.labels = label_collection

    def updata_labels(self, shot_no, labels):
        self.labels.update({"shot":int(shot_no)}, {"$set":labels}, True)


class MetaDB(object):
    def __init__(self, label_collection):
        self.labels = label_collection

    def get_labels(self, shot_no):
        result = self.labels.find_one({'shot': int(shot_no)}, {'_id': 0})
        return result


    def query(self, shot_list=None, filter=None):
        if filter is None:
            filter = {}
        if shot_list is None:
            pass
        else:
            if not ((type(shot_list) == type([])) | (type(shot_list) == type(np.array([]))) | (type(shot_list) == type(tuple([])))):
                raise ValueError("The type of shot_list isn't list or numpy.ndarray or tuple!")
            Shotslist = [int(i) for i in shot_list]
            filter.update({"shot": {"$in": Shotslist}})

        result = self.labels.find(filter, {'_id': 0})
        shots = []
        for each in result:
            shots.append(int(each['shot']))
        return shots


    def query_valid(self, shot_list=None, label_true=None, label_false=None):
        if label_true is None:
            label_true = []
        else:
            if not ((type(label_true) == type([])) | (type(label_true) == type(np.array([]))) | (type(label_true) == type(tuple([])))):
                raise ValueError("The type of label_true isn't list or numpy.ndarray or tuple!")
        if label_false is None:
            label_false = []
        else:
            if not ((type(label_false) == type([])) | (type(label_false) == type(np.array([]))) | (type(label_false) == type(tuple([])))):
                raise ValueError("The type of label_false isn't list or numpy.ndarray or tuple!")
        filter = {}
        for label_name in label_true:
            filter.update({label_name: True})
        for label_name in label_false:
            filter.update({label_name: False})
        if shot_list is None:
            pass
        else:
            if not ((type(shot_list) == type([])) | (type(shot_list) == type(np.array([]))) | (type(shot_list) == type(tuple([])))):
                raise ValueError("The type of shot_list isn't list or numpy.ndarray or tuple!")
            Shotslist = [int(i) for i in shot_list]
            filter.update({"shot": {"$in": Shotslist}})

        result = self.labels.find(filter, {'_id': 0})
        shots = []
        for each in result:
            shots.append(int(each['shot']))
        return shots


    def query_range(self, label_list, lower=None, upper=None, shot_list=None):
        if lower is None:
            lower = [None for i in range(len(label_list))]
        else:
            if len(label_list) != len(lower):
                raise ValueError("label_list and lower are not the same length!")
        if upper is None:
            upper = [None for i in range(len(label_list))]
        else:
            if len(label_list) != len(upper):
                raise ValueError("label_list and upper are not the same length!")
        filter = {}
        for i in range(len(label_list)):
            if ((lower[i] != None) & (upper[i] != None)):
                filter.update({label_list[i]:{"$gte":lower[i], "$lte":upper[i]}})
            elif ((lower[i] == None) & (upper[i] != None)):
                filter.update({label_list[i]: {"$lte": upper[i]}})
            elif ((lower[i] != None) & (upper[i] == None)):
                filter.update({label_list[i]: {"$gte": lower[i]}})
        if shot_list is None:
            pass
        else:
            if not ((type(shot_list) == type([])) | (type(shot_list) == type(np.array([]))) | (type(shot_list) == type(tuple([])))):
                raise ValueError("The type of shot_list isn't list or numpy.ndarray or tuple!")
            Shotslist = [int(i) for i in shot_list]
            filter.update({"shot": {"$in": Shotslist}})

        result = self.labels.find(filter, {'_id': 0})
        shots = []
        for each in result:
            shots.append(int(each['shot']))
        return shots


    def count_label(self, shot_list, label_list, needND=False, show=True):
        if not ((type(shot_list) == type([])) | (type(shot_list) == type(np.array([]))) | (type(shot_list) == type(tuple([])))):
            raise ValueError("The type of shot_list isn't list or numpy.ndarray or tuple!")
        if not ((type(label_list) == type([])) | (type(label_list) == type(np.array([]))) | (type(label_list) == type(tuple([])))):
            raise ValueError("The type of label_list isn't list or numpy.ndarray or tuple!")

        Shots = [int(i) for i in shot_list]
        Label_Shots = {}
        for Label_name in label_list:
            result = self.labels.find({"shot": {"$in": Shots}, Label_name: True}, {"_id": 0},)
            ValidShots = []
            for each in result:
                ValidShots.append(int(each["shot"]))
            Label_Shots.update({Label_name: ValidShots})

        complete_shots = []
        for shot in Shots:
            flag = 1
            for Label_name in label_list:
                if shot in Label_Shots[Label_name]:
                    continue
                else:
                    flag = 0
                    break
            if flag == 1:
                complete_shots.append(shot)
        disrupt_shots = []
        normal_shots = []
        disrupt_result = self.labels.find({"shot": {"$in": complete_shots}, "IsDisrupt": True}, {"_id": 0},)
        for D_each in disrupt_result:
            disrupt_shots.append(int(D_each["shot"]))
        normal_result = self.labels.find({"shot": {"$in": complete_shots}, "IsDisrupt": False}, {"_id": 0},)
        for N_each in normal_result:
            normal_shots.append(int(N_each["shot"]))

        if show:
            for Label_name in label_list:
                print("{} : ".format(Label_name) + "{}".format(len(Label_Shots[Label_name])))
            print(" ")
            print("The number of shots whose input labels are complete : {} ".format(len(complete_shots)))
            print("The number of disrupt shots : {}".format(len(disrupt_shots)))
            print("The number of normal shots  : {}".format(len(normal_shots)))
        if needND:
            return complete_shots, normal_shots, disrupt_shots
        else:
            return complete_shots

