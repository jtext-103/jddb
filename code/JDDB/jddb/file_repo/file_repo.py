import os
import h5py
import warnings

from typing import List

from ..utils import replace_pattern
from ..meta_db import meta_db


class FileRepo:
    def __init__(self, base_path: str):
        self._base_path = base_path

    @property
    def base_path(self):
        return self._base_path

    def get_all_shots(self) -> List[int]:
        # get root path from base path
        if '$' in self._base_path:
            root_path = self._base_path.split('$')[0]
        else:
            root_path = self._base_path
        # walk through root path for each file
        all_shot_list = []
        for dir_path, _, filenames in os.walk(root_path):
            for file_name in filenames:
                if file_name.endswith('.hdf5'):
                    # check if file name is a valid shot file
                    try:
                        shot_no = int(file_name.split('.')[0])
                    except ValueError:
                        continue

                    # get file and check the return path is the same as this file path
                    file_path = self.get_file(shot_no)
                    if file_path == "":
                        warnings.warn("Shot {} does not exist.".format(shot_no))
                        continue
                    if os.path.realpath(file_path) != os.path.realpath(os.path.join(dir_path, file_name)):
                        warnings.warn("Shot {} does not exist.".format(shot_no))
                        continue

                    # if all checks passed, add to all shot list
                    all_shot_list.append(shot_no)
        return all_shot_list

    def get_file(self, shot_no: int, ignore_none: bool = False) -> str:
        file_path = replace_pattern(self.base_path, shot_no)
        if os.path.exists(file_path):
            return file_path
        else:
            if ignore_none:
                return file_path
            else:
                return ""

    def get_files(self, shot_list: List[int] = None, create_empty=False) -> dict:
        file_path_dict = dict()
        if shot_list is None:
            shot_list = self.get_all_shots()
        for each_shot in shot_list:
            each_path = self.get_file(each_shot)
            if each_path == "" and create_empty:
                each_path = self.create_shot(each_shot)
            file_path_dict[each_shot] = each_path
        return file_path_dict

    @staticmethod
    def _open_file(file_path: str, mode='r'):
        try:
            hdf5_file = h5py.File(file_path, mode)
            return hdf5_file
        except OSError:
            return None

    def get_tag_list(self, shot_no: int) -> List[str]:
        file = self._open_file(self.get_file(shot_no))
        if file:
            if file.get('data') is None:
                raise KeyError("Group \"data\" does not exist.")
            else:
                if len(file.get('data').keys()) == 0:
                    file.close()
                    return list()
            tag_list = list(file.get('data').keys())
            file.close()
            return tag_list
        else:
            raise OSError('Shot {} does not exist.'.format(shot_no))

    def read_data_file(self, file_path: str, tag_list: List[str] = None) -> dict:
        data_dict = dict()
        file = self._open_file(file_path)
        if file:
            if file.get('data') is None:
                raise KeyError("Group \"data\" does not exist.")
            else:
                if len(file.get('data').keys()) == 0:
                    file.close()
                    return dict()
            if tag_list is None:
                tag_list = list(file.get('data').keys())
            for tag in tag_list:
                data_dict[tag] = file.get('data').get(tag)[()]
            file.close()
            return data_dict
        else:
            raise OSError("Invalid path given.")

    def read_data(self, shot_no: int, tag_list: List[str] = None) -> dict:
        file_path = self.get_file(shot_no)
        data_dict = self.read_data_file(file_path, tag_list)
        return data_dict

    def read_attributes(self, shot_no: int, tag: str, attribute_list: List[str] = None) -> dict:
        attribute_dict = dict()
        file = self._open_file(self.get_file(shot_no))
        if file:
            data_group = file.get('data')
            if data_group is None:
                raise KeyError("Group \"data\" does not exist.")
            else:
                dataset = data_group.get(tag)
                if dataset is None:
                    raise KeyError("{} does not exist in \"data\"".format(tag))
                else:
                    if attribute_list is None:
                        attribute_list = dataset.attrs.keys()
                    for each_attr in attribute_list:
                        attribute_dict[each_attr] = dataset.attrs.get(each_attr)
                    file.close()
            return attribute_dict
        else:
            raise OSError("Invalid path given.")

    def read_labels_file(self, file_path, label_list: List[str] = None) -> dict:
        label_dict = dict()
        file = self._open_file(file_path)
        if file:
            if file.get('meta') is None:
                raise KeyError("Group \"meta\" does not exist.")
            else:
                if len(file.get('meta').keys()) == 0:
                    file.close()
                    return dict()
            if label_list is None:
                label_list = list(file.get('meta').keys())
            for label in label_list:
                label_dict[label] = file.get('meta').get(label)[()]
            file.close()
            return label_dict
        else:
            raise OSError("Invalid path given.")

    def read_labels(self, shot_no: int, label_list: List[str] = None) -> dict:
        file_path = self.get_file(shot_no)
        label_dict = self.read_labels_file(file_path, label_list)
        return label_dict

    def remove_data_file(self, file_path: str, tag_list: List[str]):
        file = self._open_file(file_path)
        if file:
            data_group = file.get('data')
            if data_group is None:
                raise ValueError("Group \"data\" does not exist.")
            else:
                for tag in tag_list:
                    if tag not in data_group.keys():
                        warnings.warn("{} does not exist.".format(tag), category=UserWarning)
                    else:
                        file.get("data").__delitem__(tag)
            file.close()
        else:
            raise OSError("Invalid path given.")

    def remove_data(self, shot_no: int, tag_list: List[str]):
        file_path = self.get_file(shot_no)
        self.remove_data_file(file_path, tag_list)

    def remove_attributes(self, shot_no: int, tag: str, attribute_list: List[str]):
        file_path = self.get_file(shot_no)
        file = self._open_file(file_path)
        if file:
            data_group = file.get('data')
            if data_group is None:
                raise KeyError("Group \"data\" does not exist.")
            else:
                dataset = data_group.get(tag)
                if dataset is None:
                    raise KeyError("{} does not exist in \"data\"".format(tag))
                else:
                    for each_attr in attribute_list:
                        if each_attr not in dataset.attrs.keys():
                            warnings.warn("{} does not exist.".format(each_attr), category=UserWarning)
                        else:
                            dataset.attrs.__delete__(each_attr)
            file.close()
        else:
            raise OSError("Invalid path given.")

    def remove_labels_file(self, file_path: str, label_list: List[str]):
        file = self._open_file(file_path)
        if file:
            meta_group = file.get('meta')
            if meta_group is None:
                raise ValueError("Group \"meta\" does not exist.")
            else:
                for label in label_list:
                    if label not in meta_group.keys():
                        warnings.warn("{} does not exist.".format(label), category=UserWarning)
                    else:
                        file.get("meta").__delitem__(label)
            file.close()
        else:
            raise OSError("Invalid path given.")

    def remove_labels(self, shot_no: int, label_list: List[str]):
        file_path = self.get_file(shot_no)
        self.remove_labels_file(file_path, label_list)

    def create_shot(self, shot_no: int) -> str:
        file_path = self.get_file(shot_no, ignore_none=True)
        parent_dir = os.path.abspath(os.path.join(file_path,os.pardir))
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        try:
            file = h5py.File(file_path, 'x')
            file.close()
        except OSError:
            raise OSError("Shot {} already exists.".format(shot_no))
        return file_path

    def write_data_file(self, file_path: str, data_dict: dict, overwrite=False):
        file = self._open_file(file_path, 'r+')
        if file:
            if file.get('data') is None:
                warnings.warn("Group \"data\" does not exist.", category=UserWarning)
                data_group = file.create_group("data")
            else:
                data_group = file.get("data")
            tag_list = data_dict.keys()
            for tag in tag_list:
                if tag in data_group.keys():
                    if overwrite:
                        self.remove_data_file(file_path, [tag])
                    else:
                        warnings.warn("{} already exists.".format(tag), category=UserWarning)
                        continue
                data_group.create_dataset(tag, data=data_dict[tag])
            file.close()
        else:
            raise OSError("Invalid path given.")

    def write_data(self, shot_no: int, data_dict: dict, overwrite=False):
        file_path = self.get_file(shot_no)
        self.write_data_file(file_path, data_dict, overwrite)

    def write_attributes(self, shot_no: int, tag: str, attribute_dict: dict, overwrite=False):
        file = self._open_file(self.get_file(shot_no), 'r+')
        if file:
            data_group = file.get('data')
            if data_group is None:
                raise KeyError("Group \"data\" does not exist.")
            else:
                dataset = data_group.get(tag)
                if dataset is None:
                    raise KeyError("{} does not exist in \"data\"".format(tag))
                else:
                    attribute_list = attribute_dict.keys()
                    for each_attr in attribute_list:
                        if each_attr in dataset.attrs.keys():
                            if overwrite:
                                self.remove_attributes(shot_no, tag, [each_attr])
                            else:
                                warnings.warn("{} already exist.".format(each_attr), category=UserWarning)
                                continue
                        dataset.attrs.create(each_attr, attribute_dict[each_attr])
                    file.close()
        else:
            raise OSError("Invalid path given.")

    def write_label_file(self, file_path: str, label_dict: dict, overwrite=False):
        file = self._open_file(file_path, 'r+')
        if file:
            if file.get('meta') is None:
                warnings.warn("Group \"meta\" does not exist.", category=UserWarning)
                meta_group = file.create_group("meta")
            else:
                meta_group = file.get("meta")
            label_list = label_dict.keys()
            for label in label_list:
                if label in meta_group.keys():
                    if overwrite:
                        self.remove_labels_file(file_path, [label])
                    else:
                        warnings.warn("{} already exist in meat group!".format(label), category=UserWarning)
                        continue
                meta_group.create_dataset(label, label_dict[label])
            file.close()
        else:
            raise OSError("Invalid path given.")

    def write_label(self, shot_no: int, label_dict: dict, overwrite=False):
        file_path = self.get_file(shot_no)
        self.write_data_file(file_path, label_dict, overwrite)

    def sync_meta(self, meta_db: meta_db, shot_list: List[int] = None, overwrite=False):
        if shot_list is None:
            shot_list = self.get_all_shots()
        for shot in shot_list:
            label_dict = meta_db.get_labels(shot)
            del label_dict[shot]
            self.write_label(shot, label_dict, overwrite)
