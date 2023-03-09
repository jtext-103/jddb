from abc import ABCMeta, abstractmethod
from .signal import Signal
from typing import Tuple, Union


class BaseProcessor(metaclass=ABCMeta):
    """Process signals.

    Process signals according to the ``transform()`` method.

    """

    @abstractmethod
    def __init__(self):
        self.params = dict()

    @abstractmethod
    def transform(self, *signal: Signal) -> Union[Signal, Tuple[Signal, ...]]:
        """Process signal.

        This is used for subclassed processors.
        To use it, override the method.

        Args:
            signal: signal(s) to be processed.

        Returns:
            Signal: new signal processed.
        """
        raise NotImplementedError()
