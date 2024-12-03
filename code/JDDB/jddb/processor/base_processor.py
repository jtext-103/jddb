from abc import ABCMeta, abstractmethod
from .signal import Signal
from typing import Tuple, Union, Dict


class BaseProcessor(metaclass=ABCMeta):
    """Process signal(s).

    Process signals according to the ``transform()`` method.

    """

    @abstractmethod
    def __init__(self, **kwargs):
        self._init_args = kwargs
        self.params = dict()

    @abstractmethod
    def transform(self, *signal: Signal) -> Union[Signal, Tuple[Signal, ...]]:
        """Process signal(s).

        This is used for subclassed processors.
        To use it, override the method.

        Args:
            signal: signal(s) to be processed.

        Returns:
            Signal: new signal(s) processed.
        """
        raise NotImplementedError()

    def to_config(self) -> Dict:
        """Save instance information to a dictionary.

        Returns:
            A dictionary that contains type, init arguments and params of the instance.
        """
        return {
            'type': self.__class__.__name__,
            'init_args': self._init_args,
            'param': self.params
        }

    @classmethod
    def from_config(cls, processor_config: Dict):
        """Instantiate a processor from given config.

        Args:
            processor_config: A dictionary that contains init_args (necessary), type and param of the processor instance.
        Raises:
            KeyError: if 'init_args' is not in processor_config keys.
        Returns:
            Instantiated processor.
        """
        if 'init_args' not in processor_config.keys():
            raise KeyError('Config loaded does not contain \'init_args\'.')
        else:
            init_args = processor_config.get('init_args')
            processor = cls(**init_args)

        return processor