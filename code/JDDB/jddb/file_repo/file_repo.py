import os
import h5py
import warnings
from typing import List
from ..utils import replace_pattern
from ..meta_db.meta_db import MetaDB


class FileRepo:
    """
    FileRepo is used to process HDF5 files.
    base path contains template same as BM design. i.e. "\\data\\jtext\\$shot_2$XX\\$shot_1$XX\\"

    """
    def __init__(self, base_path: str):
        self._base_path = base_path
        self._data_group_name = 'data'
        self._meta_group_name = 'meta'

    @property
    def base_path(self):
        return self._base_path

    def get_file(self, shot_no: int, ignore_none: bool = False) -> str:
        """

        Get file path for one shot
        If not exist return empty string

        Args:
            shot_no: shot_no
            ignore_none: True -> even the shot file does exist, still return the file path

        Returns: file path

        """
        file_path = replace_pattern(self.base_path, shot_no)
        if os.path.exists(file_path):
            return file_path
        else:
            if ignore_none:
                return file_path
            else:
                return ""

    def get_all_shots(self) -> List[int]:
        """

        Find all shots in the base path

        Returns: shot_list

        """
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

    def create_shot(self, shot_no: int) -> str:
        """

        Create the a shot file

        Args:
            shot_no: shot_no

        Returns: file path

        """
        file_path = self.get_file(shot_no, ignore_none=True)
        parent_dir = os.path.abspath(os.path.join(file_path, os.pardir))
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        try:
            file = h5py.File(file_path, 'x')
            file.close()
        except OSError:
            raise OSError("Shot {} already exists.".format(shot_no))
        return file_path

    def get_files(self, shot_list: List[int] = None, create_empty=False) -> dict:
        """

        Get files path for a shot list

        Args:
            shot_list: shot_list
            create_empty: True -> create shot file if it does not exist before

        Returns: file_path_dict
                 --> dict{"shot_no": file_path}

        """
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
        """

        Get all the tag list of the data group in one shot file.

        Args:
            shot_no: shot_no

        Returns: tag list

        """
        file = self._open_file(self.get_file(shot_no))
        if file:
            if file.get(self._data_group_name) is None:
                raise KeyError("Group \"" + self._data_group_name + "\" does not exist.")
            else:
                if len(file.get(self._data_group_name).keys()) == 0:
                    file.close()
                    return list()
            tag_list = list(file.get(self._data_group_name).keys())
            file.close()
            return tag_list
        else:
            raise OSError('Shot {} does not exist.'.format(shot_no))

    def read_data_file(self, file_path: str, tag_list: List[str] = None) -> dict:
        """

        Read data dict from the data group in one shot file with a file path as input.

        Args:
            file_path: the file path for one shot
            tag_list: the tag list need to be read
                      if tag_list = None, read all tags

        Returns: data_dict
                 --> data_dict{"tag": data}

        """
        data_dict = dict()
        file = self._open_file(file_path)
        if file:
            if file.get(self._data_group_name) is None:
                raise KeyError("Group \"" + self._data_group_name + "\" does not exist.")
            else:
                if len(file.get(self._data_group_name).keys()) == 0:
                    file.close()
                    return dict()
            if tag_list is None:
                tag_list = list(file.get(self._data_group_name).keys())
            for tag in tag_list:
                try:
                    data_dict[tag] = file.get(self._data_group_name).get(tag)[()]
                except ValueError("{}".format(tag)):
                    raise ValueError
            file.close()
            return data_dict
        else:
            raise OSError("Invalid path given.")

    def read_data(self, shot_no: int, tag_list: List[str] = None) -> dict:
        """

        Read data dict from the data group in one shot file with a shot number as input.

        Args:
            shot_no: shot_no
            tag_list: the tag list need to be read
                      if tag_list = None, read all tags

        Returns: data_dict
                 --> data_dict{"tag": data}

        """
        file_path = self.get_file(shot_no)
        data_dict = self.read_data_file(file_path, tag_list)
        return data_dict

    def read_attributes(self, shot_no: int, tag: str, attribute_list: List[str] = None) -> dict:
        """

        Read attribute dict of one tag in one shot file.

        Args:
            shot_no: shot_no
            tag: tag
            attribute_list: the attribute list need to be read
                            if attribute_list = None, read all attributes

        Returns: attribute_dict
                 --> attribute_dict{"attribute": data}

        """
        attribute_dict = dict()
        file = self._open_file(self.get_file(shot_no))
        if file:
            data_group = file.get(self._data_group_name)
            if data_group is None:
                raise KeyError("Group \"" + self._data_group_name + "\" does not exist.")
            else:
                dataset = data_group.get(tag)
                if dataset is None:
                    raise KeyError("{} does not exist in \"" + self._data_group_name + "\"".format(tag))
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
        """

        Read label dict from the meta group in one shot file with a file path as input.

        Args:
            file_path: the file path for one shot
            label_list: the label list need to be read
                        if label_list = None, read all labels

        Returns: label_dict
                 --> label_dict{"label": data}

        """
        label_dict = dict()
        file = self._open_file(file_path)
        if file:
            if file.get(self._meta_group_name) is None:
                raise KeyError("Group \"" + self._meta_group_name + "\" does not exist.")
            else:
                if len(file.get(self._meta_group_name).keys()) == 0:
                    file.close()
                    return dict()
            if label_list is None:
                label_list = list(file.get(self._meta_group_name).keys())
            for label in label_list:
                meta_set = file.get(self._meta_group_name)
                try:
                    label_dict[label] = meta_set.get(label)[()]
                except:
                    label_dict[label] = meta_set.get(label)[:]
            file.close()
            return label_dict
        else:
            raise OSError("Invalid path given.")

    def read_labels(self, shot_no: int, label_list: List[str] = None) -> dict:
        """

        Read label dict from the meta group in one shot file with a shot number as input.

        Args:
            shot_no: shot_no
            label_list: the label list need to be read
                        if label_list = None, read all labels

        Returns: label_dict
                 --> label_dict{"label": data}

        """
        file_path = self.get_file(shot_no)
        label_dict = self.read_labels_file(file_path, label_list)
        return label_dict

    def remove_data_file(self, file_path: str, tag_list: List[str]):
        """

        Remove the datasets from the data group in one shot file with fa ile path as input.

        Args:
            file_path: the file path for one shot
            tag_list: the tag list need to be removed

        Returns: None

        """
        file = self._open_file(file_path, 'r+')
        if file:
            data_group = file.get(self._data_group_name)
            if data_group is None:
                raise ValueError("Group \"" + self._data_group_name + "\" does not exist.")
            else:
                for tag in tag_list:
                    if tag not in data_group.keys():
                        warnings.warn("{} does not exist.".format(tag), category=UserWarning)
                    else:
                        file.get(self._data_group_name).__delitem__(tag)
            file.close()
        else:
            raise OSError("Invalid path given.")

    def remove_data(self, shot_no: int, tag_list: List[str]):
        """

        Remove the datasets from the data group in one shot file with a shot number as input.

        Args:
            shot_no: shot_no
            tag_list: the tag list need to be removed

        Returns: None

        """
        file_path = self.get_file(shot_no)
        self.remove_data_file(file_path, tag_list)

    def remove_attributes(self, shot_no: int, tag: str, attribute_list: List[str]):
        """

        Remove the attribute of of one tag in one shot file.

        Args:
            shot_no: shot_no
            tag: tag
            attribute_list: the attribute list need to be removed

        Returns: None

        """
        file_path = self.get_file(shot_no)
        file = self._open_file(file_path, 'r+')
        if file:
            data_group = file.get(self._data_group_name)
            if data_group is None:
                raise KeyError("Group \"" + self._data_group_name + "\" does not exist.")
            else:
                dataset = data_group.get(tag)
                if dataset is None:
                    raise KeyError("{} does not exist in \"" + self._data_group_name + "\"".format(tag))
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
        """

        Remove labels from the meta group in one shot file with a file path as input.

        Args:
            file_path: the file path for one shot
            label_list: the label list need to be removed

        Returns: None

        """
        file = self._open_file(file_path, 'r+')
        if file:
            meta_group = file.get(self._meta_group_name)
            if meta_group is None:
                raise ValueError("Group \"" + self._meta_group_name + "\" does not exist.")
            else:
                for label in label_list:
                    if label not in meta_group.keys():
                        warnings.warn("{} does not exist.".format(label), category=UserWarning)
                    else:
                        file.get(self._meta_group_name).__delitem__(label)
            file.close()
        else:
            raise OSError("Invalid path given.")

    def remove_labels(self, shot_no: int, label_list: List[str]):
        """

        Remove labels from the meta group in one shot file with a shot number as input.

        Args:
            shot_no: shot_no
            label_list: the label list need to be removed

        Returns: None

        """
        file_path = self.get_file(shot_no)
        self.remove_labels_file(file_path, label_list)

    def write_data_file(self, file_path: str, data_dict: dict, overwrite=False):
        """

        Write a data dictionary in the data group in one shot file with a file path as input.

        Args:
            file_path: file path for one shot
            data_dict: data_dict
                       --> data_dict{"tag": data}
            overwrite: True -> remove the existed tag, then write the new one

        Returns: None

        """
        file = self._open_file(file_path, 'r+')
        if file:
            if file.get(self._data_group_name) is None:
                warnings.warn("Group \"" + self._data_group_name + "\" does not exist.", category=UserWarning)
                data_group = file.create_group(self._data_group_name)
            else:
                data_group = file.get(self._data_group_name)
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

    def write_data(self, shot_no: int, data_dict: dict, overwrite=False, create_empty=False):
        """

        Write a data dictionary in the data group in one shot file with a shot number as input.

        Args:
            shot_no: shot_no
            data_dict: data_dict
                       --> data_dict{"tag": data}
            overwrite: True -> remove the existed tag, then write the new one
            create_empty: True -> create the shot file if the shot file does not exist before

        Returns: None

        """
        if create_empty:
            file_path = self.create_shot(shot_no)
        else:
            file_path = self.get_file(shot_no)
        self.write_data_file(file_path, data_dict, overwrite)

    def write_attributes(self, shot_no: int, tag: str, attribute_dict: dict, overwrite=False):
        """

        Write attributes of one tag in one shot
        Args:
            shot_no: shot_no
            tag: tag
            attribute_dict: attribute_dict
                            --> attribute_dict{"attribute": data}
            overwrite: True -> remove the existed attribute, then write the new one

        Returns: None

        """
        file = self._open_file(self.get_file(shot_no), 'r+')
        if file:
            data_group = file.get(self._data_group_name)
            if data_group is None:
                raise KeyError("Group \"" + self._data_group_name + "\" does not exist.")
            else:
                dataset = data_group.get(tag)
                if dataset is None:
                    raise KeyError("{} does not exist in \"" + self._data_group_name + "\"".format(tag))
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
        """

        Write a label dictionary in the meta group in one shot file with a file path as input.

        Args:
            file_path: file path
            label_dict: label_dict
                        --> label_dict{"label": data}
            overwrite: True -> remove the existed label, then write the new one

        Returns: None

        """
        file = self._open_file(file_path, 'r+')
        if file:
            if file.get(self._meta_group_name) is None:
                warnings.warn("Group \"" + self._meta_group_name + "\" does not exist.", category=UserWarning)
                meta_group = file.create_group(self._meta_group_name)
            else:
                meta_group = file.get(self._meta_group_name)
            label_list = label_dict.keys()
            for label in label_list:
                if label in meta_group.keys():
                    if overwrite:
                        self.remove_labels_file(file_path, [label])
                    else:
                        warnings.warn("{} already exist in" + self._meta_group_name + "group!".format(label), category=UserWarning)
                        continue
                meta_group.create_dataset(label, data=label_dict[label])
            file.close()
        else:
            raise OSError("Invalid path given.")

    def write_label(self, shot_no: int, label_dict: dict, overwrite=False):
        """

        Write a label dictionary in the meta group in one shot file with a shot number as input.

        Args:
            shot_no: shot_no
            label_dict: label_dict
                        --> label_dict{"label": data}
            overwrite: True -> remove the existed label, then write the new one

        Returns: None

        """
        file_path = self.get_file(shot_no)
        self.write_label_file(file_path, label_dict, overwrite)

    def sync_meta(self, meta_db: MetaDB, shot_list: List[int] = None, overwrite=False):
        """

        Sync labels to the meta group of the shot file from MetaDB.

        Args:
            meta_db: initialized object of MetaDB
            shot_list: shot list
            overwrite: True -> remove the existed label, then write the new one

        Returns: None

        """
        if shot_list is None:
            shot_list = self.get_all_shots()
        for shot in shot_list:
            label_dict = meta_db.get_labels(shot)
            del label_dict['shot']
            self.write_label(shot, label_dict, overwrite)

