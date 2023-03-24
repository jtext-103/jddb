# processor

`processor` is a toolkit designed for batch processing data in a pipeline. It provides interface that can process data at different levels (signal/shot/shot set) according to different requirements. The toolkit supports HDF5 files stored in a standard format and depends on `FileRepo`.

## `signal.py`

`Signal` is the smallest unit of data processing, corresponding to a single signal in the data

    processor.Signal(
        data: numpy.ndarray,
        attributes: dict,
        tag: Optional[str] = None
    )


Arguments:

* `data`：The raw data stored in the signal, n-D `numpy.ndarray`.
* `attributes`：Metadata that describes the signal, `dict`. `attributes` must include "StartTime" (s) and "SampleRate" (Hz). If `attributes` is not specified when the instance is created, "StartTime" will be set to 0 s and "SampleRate" will be set to 1 Hz.
* `tag`：The name of the signal, `str`. Optional. Default None.

Properties:

* `data`，`attributes`，`tag`：see arguments.
* `time`：The time axis of the signal, 1D `numpy.ndarray`. Its length is the same as that of `data`, and it is calculated based on the length of data, as well as "StartTime" and "SampleRate" specified in `attributes`.

Example:

    >>> from jddb.processor import Signal
    >>> import numpy as np
    >>> example_data = np.linspace(0, 1, 1000)
    >>> example_attributes = {"StartTime": 0, "SampleRate": 1000}
    >>> example_signal = Signal(data=example_data,
                                attributes=example_attributes)
    >>> example_time = example_signal.time

## `base_processor.py`

`BaseProcessor` is an abstract class used for processing signals, which means that `BaseProcessor` cannot be instantiated on its own and must be inherited by a subclass.

    processor.BaseProcessor()

Arguments:

`BaseProcessor` itself does not have any arguments, but its subclasses can pass necessary arguments needed for data processing.

Properties:

* `params`: Parameters required for data processing, represented as a dict. It should be noted that if data is processed at the level of shot or shot set, in addition to the parameters passed in when instantiating a `BaseProcessor` subclass, the meta information for each shot can also be used in the logic of data processing.

Methods:

* `transform`
    
      transform(self, *signal: Signal) -> Union[Signal, Tuple[Signal, ...]]

`transform` is an abstract method that must be implemented by the subclasses of `BaseProcessor`. It takes in one or more signals for processing and returns one or more processed signals.

  Arguments:
  * `signal`：signal(s) to be processed.
  
  Returns:
  * `Signal`：new signal(s) processed.

Example(subclassing):

    >>>from jddb.processing import Signal, BaseProcessor
    >>>from copy import deepcopy
    >>>import numpy as np
    
    
    >>>class MultiplyProcessor(BaseProcessor):
    >>>    def __init__(self, multiplier):
    >>>        super().__init__()
    >>>        self.multiplier = multiplier
    
    >>>    def transform(self, signal: Signal) -> Signal:
    >>>        new_signal = deepcopy(signal)
    >>>        new_signal.data = new_signal.data * self.multiplier
    >>>        return new_signal

## `basic_processors/resampling_processor.py`

`ResamplingProcessor` is a subclass of `BaseProcessor`. The `ResamplingProcessor` resamples input signal to a new sample rate given.

Arguments:
* `sample_rate`: the new sample rate to resample the given signal, `float`.

Methods:
* `transform`

      transform(singal: Signal) -> Signal
  
  Args:
  * signal: the signal to be resampled, `Signal`.
  Returns:
  * `Signal`: the resampled signal.

## `basic_processors/normalization_processor.py`

`NormalizationProcessor` is a subclass of `BaseProcessor`. The `NormalizationProcessor` normalizes input signal.

Arguments:
* `std`: the std of the signal across shots, `float`.
* `mean`: the mean of the signal across shots, `float`.

Methods:
* `transform`

      transform(singal: Signal) -> Signal
  
  Args:
  * signal: the signal to be normalized, `Signal`.
  Returns:
  * `Signal`: the normalized signal.
  
  Note: The result of the normalized to signal will be clipped to [-10, 10] if beyond the range.

## `basic_processors/clip_processor.py`

`ClipProcessor` is a subclass of `BaseProcessor`. The `ClipProcessor` clips the given signal to an approximately same time range.

Arguments:
* start_time: the start time of the time axis of the clipped signal, `float`.
* end_time: the end time of the time axis of the clipped signal, `float`. Default None.
* end_time_label: the tag in `label` to find the end time of the time axis of the clipped signal, `str`. Default None. If both `end_time_tag` and `end_time` are specified, `end_time` will be ignored.

Methods:

* `transform`

      transform(signal: Signal) -> Signal
  Args:
  * signal: the signal to be clipped, `Signal`.
  
  Returns:
  * `Signal`: the clipped signal.

  Note: The start time of the clipped signal is not the same with the `start_time` passed to the `ClipProcessor`, but the first time point that is the same or after the `start_time`.

## `basic_processors/TrimProcessor.py`

`TrimProcessor` is a subclass of `BaseProcessor`. The `TrimProcessor` trims all given signals to the same length.

Methods:
* transform

      transform(*signal: Signal) -> Tuple[Signal, ...]:

  Args:
  * signal: the signals to be trimmed, `Tuple[Signal, ...]`.
  
  Returns:
  * `Tuple[Signal, ...]`: the signals trimmed with the same length.

## `shot.py`

`Shot` is a collection of `Signal`, corresponding to an HDF5 file stored in the file repo.

    processor.Shot(
        shot_no: int,
        file_repo: FileRepo
    )


Arguments:

* `shot_no`：the number of the shot, `int`.
* `file_repo`：the file repo the shot belongs to, `FileRepo`.

Attributes:

* `shot_no`，`file_repo`：see arguments, read only.
* `tags`：names of all signals stored in the `Shot` object，`set`, read only.
* `labels`: the label dictionary of the shot, `dict`.

Methods:

* `update`

      update(
        tag: str,
        signal: Signal
      )

  Add a new signal or modify an existing signal to the shot.
  
  The method DOES NOT change any file until `save()` is called.
  
  Args:
  * `tag`：name of the signal to be added or modified, `str`.
  * `signal`：the signal to be added or modified, `Signal`.


* `remove`

      remove(
        tags: Union[str, List[str]],
        keep: bool = False
      )
  
  Remove (or keep) existing signals from the shot.

  The method DOES NOT change any file until save() is called.

  Args:
  * `tags`：name(s) of the signal(s) to be removed (or kept), `str` or list of `str`.
  * `keep`：whether to remove the signals or not, `bool`. Default `False`.

  Raises:
  * `ValueError`：if any of the signal(s) is not found in the shot.

* `get`

      get(tag: str) -> Signal
  
  Get an existing signal from the shot.

  Data is read from HDF5 file ONLY when `get()` is called. Instantiating a `Shot` object will not read data to RAM.

  Args:
  * `tag`: name of the signal to be got, `str`.
  
  Returns:
  * `Signal`: the signal got.

  Raises:
  * `ValueError`：if the signal is not found in the shot.

* `process`

      process(
        processor: BaseProcessor,
        input_tags: List[Union[str, List[str]]],
        output_tags: List[Union[str, List[str]]]
      )

    Apply transformation to the signal(s) according to the processor.

    The element of input/output tags can be a string or a list of strings.

    The method DOES NOT change any file until save() is called.

    NOTE: Each element of the input and output MUST correspond to each other.

  Args:
  * `processor`: an instance of a subclassed `BaseProcessor`. The calculation is overrided in `transform()`.
  * `input_tags`: input tag(s) to be processed, `list`. Element could be `str` or list of `str`.
  * `output_tags`: output tag(s) to be processed, `list`. Element could be `str` or list of `str`.

  Raises:
  * `ValueError`: if lengths of input tags and output tags do not match.
  * `ValueError`: if lengths of output signals and output tags do not match.

* `save`

      save(
        save_repo: FileRepo = None
      )

  Save all changes done before to the specified file repo.

  Note: if the file repo give is None or shares the same base path with the origin file repo, changes will COVER the origin file. Please CHECK the new file repo to save.
  
  Args:
  * `save_repo`: file repo specified to save the shot, `FileRepo`. Default None.


Example:

    >>>from jddb.processor import Shot
    >>>from jddb.processor.basic_processor import ResamplingProcessor
    >>>from jddb.file_repo import FileRepo

    >>>base_directory = 'base/directory/to/save/hdf5'
    >>>example_file_repo = FileRepo(base_directory)
    >>>example_shot = Shot(1000000, example_file_repo)
    >>>example_shot.process(ResamplingProcessor(1000), input_tags=['\\ip', '\\bt'])
    >>>example_shot.save()

## ShotSet

ShotSet is a collection of shots, corresponding to a subset of the file repo. ShotSet supports parallel processing.

    processing.ShotSet(
      file_repo: FileRepo,
      shot_list: List[int] = None
    )

Arguments:

* `file_repo`: the file repo the shot set belongs to, `FileRepo`.
* `shot_list`: shot list within the shot set, `List[int]`. Default None. If None, return all shots in the `file_repo`.
    
Attributes:

* `file_repo`, `shot_list`: see arguments, read only.

Methods:

* `get_shot`

      get_shot(
        shot_no: int
      )
    Get an instance of Shot of given shot_no.

    Args:
    * `shot_no`: the shot to be got, `int`.

    Returns:
    * `Shot`: the instance got.

* `remove`

      remove(
        tags: List[str],
        shot_filter: List[int] = None,
        keep: bool = False,
        save_repo: FileRepo = None
      ) -> ShotSet
    Remove (or keep) existing signal(s) from the shots within the shot filter.
    
    Args:
    * `tags`: tags of signals to be removed (or kept), `List[int]`.
    * `shot_filter`: shot files to be processed within the shot set, `List[int]`. If None, process all shots within the shot set. Default None.
    * `keep`: whether to keep the tags or not, `bool`. Default False.
    * `save_repo`: file repo specified to save the shots, `FileRepo`. Default None.

    Returns:
    * `Shotset`: a new instance (or the previous instance itself) according to the base path of save_repo.

* `process`

      process(
        processor: BaseProcessor,
        input_tags: List[Union[str, List[str]]],
        output_tags: List[Union[str, List[str]]],
        shot_filter: List[int] = None,
        save_repo: FileRepo = None
      ) -> ShotSet
    Process one (or several) signal(s) of the shots within the shot filter.

    The changes WILL be saved immediately by calling this method.

    Args:
    * `processor`: an instance of a subclassed `BaseProcessor`. The calculation is overrided in `transform()`.
    * `input_tags`: input tag(s) to be processed, `list`. Element could be `str` or list of `str`.
    * `output_tags`: output tag(s) to be processed, `list`. Element could be `str` or list of `str`.
    * `shot_filter`: shot files to be processed within the shot set, `List[int]`. If None, process all shots within the shot set. Default None.
    * `save_repo`: file repo specified to save the shots, `FileRepo`. Default None.

    Returns:
    * `Shotset`: a new instance (or the previous instance itself) according to the base path of save_repo.

Example:

    >>>from jddb.processor import ShotSet
    >>>from jddb.processor.basic_processor import ResamplingProcessor
    >>>from jddb.file_repo import FileRepo
    
    >>>base_directory = 'base/directory/to/save/hdf5'
    >>>example_file_repo = FileRepo(base_directory)
    >>>example_shot_set = ShotSet(example_file_repo)
    >>>resampled_shot_set = example_shot_set.process(ResamplingProcessor(1000), input_tags=['\\ip', '\\bt'])
