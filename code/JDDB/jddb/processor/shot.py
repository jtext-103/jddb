from ..file_repo import FileRepo
from .signal import Signal
from typing import Union
from .base_processor import BaseProcessor


class Shot(object):
    """Shot object.

    An HDF5 file stored in the file repo.

    Args:
        shot_no (int): the number of the shot.
        file_repo (FileRepo): the file repo the shot belongs to.

    Attributes:
        labels (dict): the label dictionary of the shot.
    """
    def __init__(self, shot_no: int, file_repo: FileRepo):
        self._file_repo = file_repo
        self._shot_no = shot_no
        self.__new_signals = dict()
        self.__original_tags = file_repo.get_tags(self.shot_no)
        self.labels = file_repo.read_labels(self.shot_no)

    @property
    def tags(self):
        """set: tags of ALL signals in the shot, including newly added and modified. Readonly."""
        return set().union(self.__original_tags, self.__new_signals.keys())

    @property
    def shot_no(self):
        """int: the number of the shot. Readonly."""
        return self._shot_no

    @property
    def file_repo(self):
        """FileRepo: the file repo the shot belongs to. Readonly."""
        return self._file_repo

    def update(self, tag: str, signal: Signal):
        """Add a new signal or modify an existing signal to the shot.

        The method DOES NOT change any file until save() is called.

        Args:
            tag (str): name of the signal to be added or modified.
            signal (Signal): the signal to be added or modified.
        """
        self.__new_signals[tag] = signal

    def remove(self, tags: Union[str, list[str]], keep: bool = False):
        """Remove (or keep) existing signals from the shot.
        The method DOES NOT change any file until save() is called.

        Args:
            tags (Union[str, list[str]]): name(s) of the signal(s) to be removed (or kept)
            keep (bool): whether to remove the signals or not. Default False.
        Raises:
            ValueError: if any of the signal(s) is not found in the shot.
        """
        if isinstance(tags, str):
            tags = [tags]
        tags = set(tags)
        for tag in tags:
            if tag not in self.tags:
                raise ValueError("{} is not found in data.".format(tag))
        if keep:
            drop_tags = self.tags.difference(tags)
        else:
            drop_tags = tags
        for tag in drop_tags:
            if tag in self.__original_tags:
                self.__original_tags.remove(tag)
            if tag in self.__new_signals.keys():
                del self.__new_signals[tag]

    def get(self, tag: str) -> Signal:
        """Get an existing signal from the shot.

        Args:
            tag (str): name of the signal to be got.
        Returns:
            Signal: the signal got.
        Raises:
            ValueError: if the signal is not found in the shot.
        """
        if tag in self.__new_signals.keys():
            return self.__new_signals[tag]
        elif tag in self.__original_tags:
            return Signal(data=self.file_repo.read_data(self.shot_no, [tag])[tag],
                          attributes=self.file_repo.get_attr(self.shot_no))
        else:
            raise ValueError("{} is not found in data.".format(tag))

    def process(self, processor: BaseProcessor, input_tags: list[Union[str, list[str]]],
                output_tags: list[Union[str, list[str]]]):
        """Process one (or multiple) signals of the shot.

        Apply transformation to the signal(s) according to the processor.
        The element of input/output tags can be a string or a list of strings.
        The method DOES NOT change any file until save() is called.

        Note: Each element of the input and output MUST correspond to each other.

        Args:
            processor (BaseProcessor): an instance of a subclassed BaseProcessor. The calculation is overrided in transform().
            input_tags (list[Union[str, list[str]]]): input tag(s) to be processed.
            output_tags (list[Union[str, list[str]]]): output tag(s) to be processed.
        Raises:
            ValueError: if lengths of input tags and output tags do not match.
            ValueError: if lengths of output signals and output tags do not match.
        """
        if len(input_tags) != len(output_tags):
            raise ValueError("Lengths of input tags and output tags do not match.")
        processor.params.update(self.labels)
        processor.params.update({"Shot": self.shot_no})

        for i_tag, o_tag in zip(input_tags, output_tags):
            if isinstance(i_tag, str):
                new_signal = processor.transform(self.get(i_tag))
            else:
                new_signal = processor.transform(*[self.get(each_tag) for each_tag in i_tag])

            if isinstance(o_tag, str) and isinstance(new_signal, Signal):
                self.update(o_tag, new_signal)

            elif isinstance(o_tag, list) and isinstance(new_signal, list):
                if len(o_tag) != len(new_signal):
                    raise ValueError("Lengths of output tags and signals do not match!")
                for idx, each_signal in enumerate(new_signal):
                    self.update(o_tag[idx], each_signal)
            else:
                raise ValueError("Lengths of output tags and signals do not match!")

    def save(self, save_repo: FileRepo = None):
        """Save the shot to specified file repo.

        Save all changes done before to disk space.
        Note: if the file repo give is None or shares the same base path with the origin file repo,
        changes will COVER the origin file. Please CHECK the new file repo to save.

        Args:
            save_repo (FileRepo): file repo specified to save the shot. Default None.
        """
        if save_repo is not None and (save_repo.base_path != self.file_repo.base_path):
            output_path = save_repo.create_shot(self.shot_no)

            for tag in self.tags:
                signal = self.get(tag)
                save_repo.write_data_file(output_path, signal.data, signal.attributes)
            save_repo.write_label_file(output_path, self.labels)

        else:
            existing_tags = self.file_repo.get_tags(self.shot_no)
            tags_to_remove = [r_tag for r_tag in existing_tags if r_tag not in self.tags]
            tags_to_update = list(set(self.__new_signals.keys()) & set(existing_tags))

            self.file_repo.remove_data(self.shot_no, tags_to_remove+tags_to_update)

            for w_tag, w_signal in self.__new_signals.items():
                self.file_repo.write_data(self.shot_no, w_signal.data, w_signal.attributes)

            self.file_repo.write_label(self.shot_no, self.labels)
