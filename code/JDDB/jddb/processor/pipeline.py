from ..processor import Shot, ShotSet, BaseProcessor
from ..file_repo import FileRepo
import yaml
from typing import List, Union, Dict
import logging
from functools import partial
import platform
import multiprocessing as mp
from multiprocessing import Queue
from logging.handlers import QueueHandler, QueueListener
from logging import FileHandler

class Step:
    def __init__(self, processor: BaseProcessor, input_tags: List[Union[str, List[str]]],
                 output_tags: List[Union[str, List[str]]]):
        pass

    def execute(self, shot: Shot):
        pass

    def to_config(self) -> Dict:
        pass

    def to_yaml(self, filepath: str = 'step_config.yaml'):
        pass

    @classmethod
    def from_config(cls, step_config: Dict):
        pass

    @classmethod
    def from_yaml(cls, filepath: str):
        pass

class Pipeline:
    def __init__(self, steps: List[Step] = None):
        pass

    def add_step(self, step: Step):
        pass

    def to_config(self) -> Dict:
        pass

    def to_yaml(self, filepath: str = 'pipeline_config.yaml'):
        pass

    @classmethod
    def from_config(cls, pipeline_config: Dict):
        pass

    @classmethod
    def from_yaml(cls, filepath: str):
        pass

    def process_by_shot(self, shot: Shot, save_repo: FileRepo = None):
        pass

    def process_by_shotset(self, shotset: ShotSet, processes: int = mp.cpu_count() - 1,
                           shot_filter: List[int] = None, save_repo: FileRepo = None):
        pass

    def _subprocess_task_in_shotset(self, shot_no: int, queue: Queue, shot_set: ShotSet, save_repo: FileRepo = None):
        pass