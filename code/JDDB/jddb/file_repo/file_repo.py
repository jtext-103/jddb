import numpy as np


class FileRepo:
    def __init__(self, base_path: str):
        self._base_path = base_path
        pass

    @property
    def base_path(self):
        return self._base_path

    @property
    def shot_list(self) -> list[int]:
        pass

    def get_file(self, shot_no: int) -> str:
        pass

    def get_files(self, shot_list=None, create_empty=False) -> list[str]:
        pass

    def get_tags(self, shot_no: int) -> list[str]:
        pass

    def get_attr(self, shot_no: int) -> dict:
        pass

    def read_data(self, shot_no: int, tags: list[str]) -> dict:
        pass

    def read_data_file(self, file_path: str, tags: list[str]) -> dict:
        pass

    def read_labels(self, shot_no: int) -> dict:
        pass

    def create_shot(self, shot_no: int) -> str:
        pass

    def write_label(self, shot_no: int, labels: dict):
        pass

    def write_label_file(self, file_path: str, labels: dict):
        pass

    def write_data(self, shot_no: int, data: np.ndarray, attributes: dict):
        pass

    def write_data_file(self, file_path: str, data: np.ndarray, attributes: dict):
        pass

    def remove_data(self, shot_no: int, tags: list[str]):
        pass

    def remove_data_file(self, file_path: str, tags: list[str]):
        pass
