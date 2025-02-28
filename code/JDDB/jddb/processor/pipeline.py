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
        self.steps = steps if steps else list()

    def add_step(self, step: Step):
        self.steps.append(step)

    def to_config(self) -> Dict:
        pipeline_config = dict()
        for i, step in enumerate(self.steps):
            pipeline_config[f'Step_{i + 1}'] = step.to_config()
        return pipeline_config

    def to_yaml(self, filepath: str = 'pipeline_config.yaml'):
        pipeline_config = self.to_config()
        with open(filepath, 'w') as file:
            yaml.dump(pipeline_config, file)

    @classmethod
    def from_config(cls, pipeline_config: Dict, processor_registry: Dict):
        steps = list()
        for i in range(len(pipeline_config)):
            steps.append(Step.from_config(pipeline_config[f'Step_{i + 1}'], processor_registry))
        return cls(steps=steps)

    @classmethod
    def from_yaml(cls, filepath: str, processor_registry: Dict):
        with open(filepath, 'r') as file:
            pipeline_config = yaml.safe_load(file)

        return cls.from_config(pipeline_config=pipeline_config, processor_registry=processor_registry)

    def process_by_shot(self, shot: Shot, save_repo: FileRepo = None, save_updated_only: bool = False):
        logger = logging.getLogger('single_process')
        logger.setLevel(logging.ERROR)
        if not logger.hasHandlers():
            file_handler = logging.FileHandler('./process_exceptions.log', mode="a")
            file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            logger.addHandler(file_handler)

        for i, step in enumerate(self.steps):
            try:
                step.execute(shot)
            except Exception as e:
                error_message = (
                    f"Shot No: {shot.shot_no}\n"
                    f"Pipeline Step: {i + 1}\n"
                    f"Processor Type: {type(step.processor).__name__}\n"
                    f"Input Signals: {step.input_tags}\n"
                    f"Output Signals: {step.output_tags}\n"
                    f"Error Message: {e}\n"
                )
                logger.error(msg=error_message, exc_info=None)

        shot.save(save_repo, save_updated_only=save_updated_only)

    def process_by_shotset(self, shotset: ShotSet, processes: int = mp.cpu_count() - 1,
                           shot_filter: List[int] = None, save_repo: FileRepo = None, save_updated_only: bool = False):
        """Process all shots in a ShotSet."""
        if shot_filter is None:
            shot_filter = shotset.shot_list
        if processes:
            if platform.system() == 'Linux':
                pool = mp.get_context('spawn').Pool(processes=processes)
            else:
                pool = mp.Pool(processes=processes)
            logger = logging.getLogger('multi_process')
            logger.setLevel(logging.ERROR)
            queue = mp.Manager().Queue(-1)
            queue_handler = QueueHandler(queue)
            logger.addHandler(queue_handler)
            listener = QueueListener(
                queue,
                FileHandler('./process_exceptions.log', mode="a")
            )
            listener.start()
            pool.map(partial(self._subprocess_task_in_shotset, queue=queue, shot_set=shotset, save_repo=save_repo,
                             save_updated_only=save_updated_only), shot_filter)
            pool.close()
            pool.join()
            listener.stop()
        else:
            for shot_no in shot_filter:
                shot = shotset.get_shot(shot_no)
                self.process_by_shot(shot, save_repo=save_repo)

    def _subprocess_task_in_shotset(self, shot_no: int, queue: Queue, shot_set: ShotSet, save_repo: FileRepo = None,
                                    save_updated_only: bool = False):
        single_shot = shot_set.get_shot(shot_no)
        for i, step in enumerate(self.steps):
            try:
                step.execute(single_shot)
            except Exception as e:
                error_message = (
                    f"Shot No: {shot_no}\n"
                    f"Pipeline Step: {i + 1}\n"
                    f"Processor Type: {type(step.processor).__name__}\n"
                    f"Input Signals: {step.input_tags}\n"
                    f"Output Signals: {step.output_tags}\n"
                    f"Error Message: {e}\n"
                )
                queue.put(
                    logging.LogRecord('multi_process', logging.ERROR, '', 0, error_message, exc_info=None, args=None))

        single_shot.save(save_repo=save_repo, save_updated_only=save_updated_only)
