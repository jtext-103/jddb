# extarctor

`extarctor` BaseExtractor is a toolkit designed for extracting features in a pipeline. The toolkit supports HDF5 files stored in a standard format and depends on `processor`.

## `base_extarctor.py`

`BaseExtractor` is the smallest unit of data processing, corresponding to a single signal in the data

    extarctor.BaseExtractor(
        config_file_path: str
    )

Arguments:

* `config_file_path`：Path to the configuration file, which contains parameters required for feature extraction.

Properties:

* `config_file_path`：Stores the path of the configuration file,`str`.
* `config`：A dictionary containing the extracted configuration settings, `dict`.

Methods:

* `load_config`
    
      load_config(self) -> dict

`load_config` Loads the configuration file from the specified path.

  Returns:
  * `dict`：The loaded configuration as a dictionary.

* `extract_steps`

      extract_steps(self) -> List[Step]

    Returns:
    * `List[Step]`：A list of Step objects representing the extraction steps.

* `make_pipeline`

      make_pipeline(self) -> Pipeline

     Returns:
    * `jddb.processor.Pipeline`：This method must be implemented by subclasses to create a specific processing pipeline.

Example:
    
    >>> from jddb.processor import extractors
    >>>
    >>> # make a custom extractor
    >>> class CustomExtractor(BaseExtractor):
    >>> def __init__(self, config_file_path: str):
    >>>     self.config_file_path = config_file_path
    >>>     self.config = self.load_config()
    >>> def extract_steps(self):
    >>>     return [Step(processor=Resample(1000), input_tags=["\\ip"], output_tags=["\\ip"])]]
    >>> def make_pipeline(self):
    >>>     return Pipeline(self.extract_steps())
    >>>
    >>> # Initialize an extractor with a config file
    >>> extractor = CustomExtractor("config.json")
    >>> # Loaded configuration
    >>> extractor_config = extractor.config  
    >>> extractor_steps = extractor.extract_steps() 
    >>> # Created pipeline
    >>> extractor_pipeline = extractor.make_pipeline()  
