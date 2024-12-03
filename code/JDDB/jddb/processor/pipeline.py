from ..processor import Shot, ShotSet, BaseProcessor, Signal
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
        assert len(input_tags) == len(output_tags), "Lengths of input tags and output tags do not match."
        self.processor = processor
        self.input_tags = input_tags
        self.output_tags = output_tags

    def execute(self, shot: Shot):
        self.processor.params.update(shot.labels)
        self.processor.params.update({"Shot": shot.shot_no})

        for i_tag, o_tag in zip(self.input_tags, self.output_tags):
            if isinstance(i_tag, str):
                new_signal = self.processor.transform(shot.get_signal(i_tag))
            else:
                new_signal = self.processor.transform(*[shot.get_signal(each_tag) for each_tag in i_tag])

            if isinstance(o_tag, str) and isinstance(new_signal, Signal):
                shot.update_signal(o_tag, new_signal)

            elif isinstance(o_tag, list) and isinstance(new_signal, tuple):
                if len(o_tag) != len(new_signal):
                    raise ValueError("Lengths of output tags and signals do not match!")
                for idx, each_signal in enumerate(new_signal):
                    shot.update_signal(o_tag[idx], each_signal)
            else:
                raise ValueError("Lengths of output tags and signals do not match!")

        return shot

    def to_config(self) -> Dict:
        step_config = {
            'processor_type': type(self.processor).__name__,
            'input_tags': self.input_tags,
            'output_tags': self.output_tags,
            'params': self.processor.params,
            'init_args': self.processor.to_config().get('init_args')
        }
        return step_config

    def to_yaml(self, filepath: str = 'step_config.yaml'):
        config = self.to_config()
        with open(filepath, 'w') as file:
            yaml.dump(config, file)

    @classmethod
    def from_config(cls, step_config: Dict, processor_registry: Dict):
        processor_type = step_config['processor_type']
        if processor_type not in processor_registry:
            raise ValueError(f"Processor type '{processor_type}' not found in the registry.")

        # Dynamically initialize processor using saved arguments
        processor_cls = processor_registry[processor_type]
        init_args = step_config.get('init_args')
        print(processor_cls)
        print(init_args)
        processor = processor_cls(**init_args)
        processor.params = step_config.get('params')

        return cls(processor, step_config['input_tags'], step_config['output_tags'])

    @classmethod
    def from_yaml(cls, filepath: str, processor_registry: Dict):
        with open(filepath, 'r') as file:
            step_config = yaml.safe_load(file)

        return cls.from_config(step_config=step_config, processor_registry=processor_registry)

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