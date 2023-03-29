import os
import numpy as np
from pymongo import MongoClient


class MetaDB(object):
    """
    A class that deals with MetaDB.

    Attributes:
        connection_str (dict): Information about connecting to MetaDB.
        collection (str): The name of the collection that needs to be connected.
    """
    def __init__(self, connection_str, collection):
        """
        Constructs a new MetaDB object.

        Args:
            connection_str (dict): Information about connecting to MetaDB.
            collection (str): The name of the collection that needs to be connected.
        """
        self.labels = None
        self.client = None
        self.connect(connection_str, collection)

    def connect(self, connection_str, collection):
        """
        Connect to MetaDB before any other action.

        Args:
            connection_str (dict): Information about connecting to MetaDB.
            collection (str): The name of the collection that needs to be connected.

        Return:
            pymongo.collection.Collection: Collection of the MetaDB.
        """
        self.client = MongoClient(connection_str["host"], int(connection_str["port"]))
        db = self.client[connection_str["database"]]
        if ("username" in connection_str.keys()) & ("password" in connection_str.keys()):
            db.authenticate(connection_str["username"], connection_str["password"])
        labels = db[collection]
        self.labels = labels

    def disconnect(self):
        """
        Disconnect from MetaDB after any other action.

        Args:
            None

        Return:
            None
        """
        if self.client is not None:
            self.client.close()

    def update_labels(self, shot_no, labels):
        """
        Update or modify the meta of a shot in MetaDB.

        Args:
            shot_no (int or string): The shot number whose meta you want to update or modify.
            labels (dict): The meta contents you want to update or modify.

        Return:
            None
        """
        self.labels.update({"shot":int(shot_no)}, {"$set":labels}, True)


    
    def get_labels(self, shot_no):
        """
        Get all meta of the input shot.

        Args:
            shot_no (int or string): The shot number whose meta you want to get.

        Return:
            dict: All meta content of the inuput shot.
        """
        result = self.labels.find_one({'shot': int(shot_no)}, {'_id': 0})
        return result


    def query(self, shot_list=None, filter=None):
        """
        Query the shots that meet the filter conditions within the given shot number range.

        Args:
            shot_list (list): The queried range of shot numbers.
            filter (dict): The filter condition for the query.

        Return:
            list: Shot numbers that meet the filter condition.
        """
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
        """
        For labels whose information stored in the database is True or False, return shot numbers that meet the filter condition.

        Args:
            shot_list (list): The queried range of shot numbers.
            label_true (list): The returned shots must satisfy that all labels in the label_true are True.
            label_false (list): The returned shots must satisfy that all labels in the label_false are False.

        Return:
            list: Shot numbers that meet the filter condition.
        """
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


    def query_range(self, label_list, lower_limit=None, upper_limit=None, shot_list=None):
        """
        For labels with numeric values stored in the database, return shot numbers that meet the filter condition.

        Args:
            label_list (list): A list of labels that store information as numeric values.
            lower_limit (list): List of lower limit values. ">=".
            upper_limit (list): List of upper limit values. "<=".
            shot_list (list): The queried range of shot numbers.

        Return:
            list: Shot numbers that meet the filter condition.
        """
        if lower_limit is None:
            lower_limit = [None for i in range(len(label_list))]
        else:
            if len(label_list) != len(lower_limit):
                raise ValueError("label_list and lower_limit are not the same length!")
        if upper_limit is None:
            upper_limit = [None for i in range(len(label_list))]
        else:
            if len(label_list) != len(upper_limit):
                raise ValueError("label_list and upper_limit are not the same length!")
        filter = {}
        for i in range(len(label_list)):
            if ((lower_limit[i] != None) & (upper_limit[i] != None)):
                filter.update({label_list[i]:{"$gte":lower_limit[i], "$lte":upper_limit[i]}})
            elif ((lower_limit[i] == None) & (upper_limit[i] != None)):
                filter.update({label_list[i]: {"$lte": upper_limit[i]}})
            elif ((lower_limit[i] != None) & (upper_limit[i] == None)):
                filter.update({label_list[i]: {"$gte": lower_limit[i]}})
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


    def count_label(self, shot_list, label_list, need_nd=False, show=True):
        """
        Count and display the number of shots available in the given shot number list for each given diagnosis;
        Finally, count and display the number of shots available for all given diagnostic signals in the given
        shot number list, as well as the number of non-disruption and disruption shots. Returns the list of available
        shots for diagnosis.

        Args:
            shot_list (list): The queried range of shot numbers.
            label_list (list): The queried label names.
            need_nd (bool):  Whether to divide the returned shots into disruption shots and non-disruption shots, and return the disruption shots and non-disruption shots.
            show (bool): Whether to display statistical results.

        Return:
            list: When need_nd=False, only return one list, which is the list of shots available for all given diagnostic signals.
                  When need_nd=True, return three lists, which are the list of shots available for all given diagnostic signals, list of non-disruption shots, and list of disruption shots.
        """
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
        if need_nd:
            return complete_shots, normal_shots, disrupt_shots
        else:
            return complete_shots

