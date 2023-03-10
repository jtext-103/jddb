from __future__ import annotations
from ..file_repo import FileRepo
from .base_processor import BaseProcessor
from .shot import Shot
from typing import Union


class ShotSet(object):
    """ShotSet object.
    A subset of file repo. Support pipeline of data processing.

    Args:
        file_repo (FileRepo): the file repo the shot set belongs to.
        shot_list (list[int]): shot list within the shot set.
    """
    def __init__(self, file_repo: FileRepo, shot_list: list[int] = None):
        self._file_repo = file_repo
        if shot_list is None:
            self._shot_list = file_repo.shot_list
        else:
            self._shot_list = shot_list

    @property
    def file_repo(self):
        """FileRepo: the file repo the shot set belongs to. Readonly."""
        return self._file_repo

    @property
    def shot_list(self):
        """list (int): shot list within the shot set. Readonly."""
        return self._shot_list

    def get_shot(self, shot_no: int):
        """Get an instance of Shot of given shot_no.

        Args:
            shot_no (int): the shot to be got.
        Returns:
            Shot: the instance got.
        """
        return Shot(shot_no, self.file_repo)

    def remove(self, tags: list[str], shot_filter: list[int] = None, keep: bool = False, save_repo: FileRepo = None) -> ShotSet:
        """Remove (or keep) existing signal(s) from the shots within the shot filter.

        The changes removed WILL be saved immediately by calling this method.

        Args:
            tags (list[str]): tags of signals to be removed (or kept).
            shot_filter (list[int]): shot files to be processed within the shot set. If None, process all shots within the shot set. Default None.
            keep (bool): whether to keep the tags or not. Default False.
            save_repo (FileRepo): file repo specified to save the shots. Default None.
        Returns:
            ShotSet: a new instance (or the previous instance itself) according to the base path of save_repo.
        """
        if shot_filter is None:
            shot_filter = self.shot_list
        for each_shot in shot_filter:
            shot = Shot(each_shot, self.file_repo)
            shot.remove(tags, keep)
            shot.save(save_repo)
        if save_repo is None and shot_filter == self.shot_list:
            return self
        else:
            return ShotSet(save_repo, shot_filter)

    def process(self, processor: BaseProcessor, input_tags: list[Union[str, list[str]]], output_tags: list[Union[str, list[str]]],
                shot_filter: list[int] = None, save_repo: FileRepo = None) -> ShotSet:
        """Process one (or several) signal(s) of the shots within the shot filter.

        The changes WILL be saved immediately by calling this method.

        Args:
            processor (BaseProcessor): an instance of a subclassed BaseProcessor.
            input_tags (list[Union[str, list[str]]]): input tag(s) to be processed.
            output_tags (list[Union[str, list[str]]]): output tag(s) to be processed.
            shot_filter (list[int]): shot files to be processed within the shot set. If None, process all shots within the shot set. Default None.
            save_repo (FileRepo): file repo specified to save the shots. Default None.
        Returns:
            ShotSet: a new instance (or the previous instance itself) according to the base path of save_repo.
        """
        if len(input_tags) != len(output_tags):
            raise ValueError("Lengths of input tags and output tags do not match.")
        if shot_filter is None:
            shot_filter = self.shot_list
        for each_shot in shot_filter:
            shot = Shot(each_shot, self.file_repo)
            shot.process(processor, input_tags, output_tags)
            shot.save(save_repo)
        if save_repo is None and shot_filter == self.shot_list:
            return self
        else:
            return ShotSet(save_repo, shot_filter)
