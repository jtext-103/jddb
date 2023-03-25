# How to use Performance 

**`performance`** is the subpackages used to evaluate model performance shot by shot.

It has two class named **`Result`** and **`Report`**.

  
## `result.py`

`Result` is used to evaluate model performance shot by shot. 
When initialized, the table in the csv file is read, the shot_no in the first row of the table is set to -10 to store the threshold 
        

Arguments:

* `csv_path`: the csv file path incoming to the instantiated Result


Properties:

* `IS_DISRUPT`, `DOWN_TIME` is the meta of a shot.
* `PREDICTED_DISRUPTION_H`, `PREDICTED_DISRUPTION_TIME_H`, `ACTUAL_DISRUPTION_H`, `ACTUAL_DISRUPTION_TIME_H`,
  `WARNING_TIME_H`, `TRUE_POSITIVE_H`, `FALSE_POSITIVE_H`, `TRUE_NEGATIVE_H`, `FALSE_NEGATIVE_H`, `TARDY_ALARM_THRESHOLD_H`,
  `LUCKY_GUESS_THRESHOLD_H` are elements of the header of a dataframe.
* `tpr`, `fpr`, `accuracy`, `precision`, `recall`, `confusion_matrix`, `average_warning_time`, `median_warning_time`,
* `tardy_alarm_threshold`, `lucky_guess_threshold` are minimum threshold and maximum threshold that a predicted disruptive time ahead of the actual disruptive time.
* `csv_path`is the csv file path incoming to the instantiated Result.
* `result` is a dataframe read from csv_path.
* `header` is the header of the property `result`.
* `y_true` is a label that actual disruption is True or False, as the parameter passing into the interface of sklearn to calculate the model performance evaluation.
* `y_pred` is a label that predicted disruption is True or False, as the parameter passing into the interface of sklearn to calculate the model performance evaluation.
* `ignore_thresholds` is used to choose whether to consider thresholds to constrain the correct disruptive warning time.


Methods:

* `read`

      read()
    Read in the header information of all shots set in property `result`, read in property `tardy_alarm_threshold`,`lucky_guess_threshold`.


* `save`

      save()
    Save calculated table in the disk, all processing is only stored in memory before calling save().


* `add`

      add(
        shot_list: List[int],
        predicted_disruption: List[int],
        predicted_disruption_time: List[float]
      )
  
    Add the predicted information `shot_no`,  `predicted_disruption`, `predicted_disruption_time` of shot(s) from the table of property `result`.
    If the shot number already exists, the value in the corresponding position of the row will be overwritten, 
    and if there are no duplicate shot numbers, it will be added to the last row of the table.
    
    Args:
    * `shot_list`: shot number of shots to be added, `List[int]`.
    * `predicted_disruption`: predicted disruption label of shots with a value of 0 or 1, `List[int]`. 
    The index of predicted_disruption corresponds to shot_list.
    * `predicted_disruption_time`:predicted disruption time of shots with the unit seconds, `List[float]`. 
    The index of predicted_disruption corresponds to shot_list.


* `remove`

      remove(
        shot_list: List[int]
      )

    Remove the information of existing shot(s) from the table of property `result`. 
    There are no changes to the table if the shot number doesn't exist.
    
    Args:
    * `shot_list`: shot number of shots to be removed , `List[int]`.


* `get_all_truth_from_metadb`

      get_all_truth_from_metadb(
        meta_db: MetaDB
      )
  
    Add the actual information from Meta DB, reading in `actual_disruption_time` and `actual_disruption_time` of existing shot(s) from the table of property `result`.
    If the shot number already exists, the value in the corresponding position of the row will be overwritten, 
    and if there are no duplicate shot numbers, it will be added to the last row of the table.
    
    Args:
    * `meta_db` is a initialized a class `MetaDB` from the module `meta_db.py`.


* `get_all_truth_from_file_repo`

      get_all_truth_from_file_repo(
        file_repo: FileRepo
      )
    
    Add the actual information from FileRepo, reading in `actual_disruption_time` and `actual_disruption_time` of existing shot(s) from the table of property `result`.
    If the shot number already exists, the value in the corresponding position of the row will be overwritten, 
    and if there are no duplicate shot numbers, it will be added to the last row of the table.
    
    Args:
    
    * `file_repo` is a initialized a class `FileRepo` from the module `file_repo.py`.


* `get_all_shots`

      get_all_shots( 
        include_all=True
      )

    get all shot_list 

    Args:
    * `include_all` is to choose whether to include raw that without actual information,

    Returns:
    * `shot_list`: a list of shot number.
    if include_all=True, return all shot of dataframe.
    if include_all=False, return shot list except that the value of actual_disruption is `np.nan`.


* `calc_metrics`

      calc_metrics()

    this function should be called before setting property `tardy_alarm_threshold` and `lucky_guess_threshold`,
    Calculate the following parameters row by row and fill in the corresponding positions of property result: 
    `warning_time`, `true_positive`, `false_positive`, `true_negative`, `false_negative` and property `y_pred`,
    `y_true`, `confusion_matrix`, `tpr`, `fpr`, `accuracy`, `precision`, `recall`, `average_warning_time`, `median_warning_time`.

  
* `get_shot_result`

      get_shot_result(
      pred,
      pred_time,
      truth,
      truth_downtime,
      ignore_threshold=False
      )

    This function called by calc_metrics(),
    computing `warning_time`, `true_positive`, `false_positive`, `true_negative`, `false_negative`.

        Args:
    * `ignore_threshold` is to choose whether to consider `tardy_alarm_threshold`, `lucky_guess_threshold`,
    Defalt False is to consider those two.

    Returns:
    * `tp`: true_positive.
    * `fp`: false_positive
    * `tn`: true_negative
    * `fn`: false_negative
    * `warning_time`: truth_downtime - pred_time, unit: s, if undisrupted or the predicted disruptive was wrong, set -1.


* `get_average_warning_time`

      get_average_warning_time()
    
    This function called by calc_metrics(),
    computing value of average warning_time with the true disruptive shots

    Returns:
    * `average_warning_time`: mean of warning time, unit: s


* `get_median_warning_time`

      get_median_warning_time()

    This function called by calc_metrics(),
    computing value of median warning time with the true disruptive shots

    Returns:
    * `median_warning_time`: median of warning time, unit: s


* `get_confusion_matrix`

      get_confusion_matrix()
    
    This function called by calc_metrics(),
    computing value of get_confusion_matrix from the shots with actual information.

    Returns:
    * `confusion_matrix`: a dimension of 2*2 matrix equals [[`true_positive`, `false_positive`],
                                                             [`true_negative`, `false_negative`]]


* `get_ture_positive_rate`

      get_ture_positive_rate()

    This function called by calc_metrics(),
    computing value of ture positive rate and ture positive rate from the shots with actual information.

    Returns:
    * `true_positive_rate`
    * `false_positive_rate`


* `get_accuracy`

      get_accuracy()

    This function called by calc_metrics(),
    computing value of accuracy from the shots with actual information.

    Returns:
    * `accuracy`: (tp + tn)/(tp + tn + fp + fn)
  

* `get_precision`

      get_precision()
    
    This function called by calc_metrics(),
    computing value of precision from the shots with actual information.

    Returns:
    * `precision`: tp /(tp + fp)


* `get_recall`

      get_recall()

    This function called by calc_metrics(),
    computing value of recall from the shots with actual information.

    Returns:
    * `recall`: tp / positive predicted


* `plot_warning_time_histogram`

      plot_warning_time_histogram( 
        time_bins: List[float],
        file_path=None
      )

    this function should be called before call self.calc_metrics().
    Plot a column chart, the x-axis is time range, the y-axis is the number of shot during that waring time threshold.

    Args:
    *  `time_bins` the end point of time range in x-axis, List[float].
    *  `file_path` is the path to save the picture, if file_path=None, just show in memory.

      
* `plot_accumulate_warning_time`

      plot_accumulate_warning_time( 
        file_path=None
      )

    this function should be called before call self.calc_metrics().
    Plot a line chart, the x-axis is time range, the y-axis is the percentage of the number of shot during that waring time period.
    the total number of shot equals to false positive number plus true positive number

    Args:
    * `file_path` is the path to save the picture, if file_path=None, just show in memory.


Example:

    >>>from jddb.performance import Result
    >>>from jddb.meta_db import MetaDB
    >>>from jddb.file_repo import FileRepo
    >>>base_directory = 'base/directory/to/save/hdf5'
    >>>example_file_repo = FileRepo(base_directory)
    >>>example_shot_set = ShotSet(example_file_repo)
    >>>csv_path = 'base/directory/to/save/test_result.csv'
    >>>test_result = Result(csv_path)
    >>>test_result.lucky_guess_threshold = .3 
    >>>test_result.tardy_alarm_threshold = .005
    >>>result.add(shot_no=[123012], predicted_disruption=[1], predicted_disruption_time=[0.1])
    >>>test_result.calc_metrics()
    >>>test_result.remove([100563])
    >>>test_result.warning_time_histogram([-1, 0, 0.01, 0.02, 0.03])
    >>>test_result.save()

  
## `report.py`

`Report` is used to evaluate models performance. When initialized, the table in the csv file is read.
        

Arguments:

* `csv_path`: the csv file path incoming to the instantiated Report


Properties:

* `MODEL_NAME`, `ACCURACY`, `PRECISION`, `RECALL`,`FPR`, `TPR`, `TP`, `FN`, `FP`, `TN` are elements of the header of a dataframe.
* `csv_path`is the csv file path incoming to the instantiated Report.
* `report` is a dataframe read from csv_path.
* `header` is the header of the property `report`.


Methods:

* `read`

      read()
    Read in the header information of all models set in property `report`.


* `save`

      save()
    Save calculated table in the disk, all processing is only stored in memory before calling save().


* `add`

      add(
        result,
        model_name: str
      )
  
    Add a name of model and those following performance evaluation to the table of property `result`:
   `MODEL_NAME`, `ACCURACY`, `PRECISION`, `RECALL`,`FPR`, `TPR`, `TP`, `FN`, `FP`, `TN`
    If the model already exists, the value in the corresponding position of the row will be overwritten, 
    and if there are no duplicate model, it will be added to the last row of the table.
    
    Args:
    * `model_name` is a list of name of model(s) to be added, `List[int]`.
    * `result` is an initialized `Result`, a class from the module `result.py`

  
* `remove`

      remove(
        shot_list: List[int]
      )

    Remove the information of existing shot(s) from the table of property `result`. 
    There are no changes to the table if the name of model doesn't exist.
    
    Args:
    * `model`: a list of name of model(s) to be removed , `List[int]`.


* `plot_roc`

      plot_roc(
        roc_file_path =None
      )

    Plot a roc curve, the x-axis is false positive rate, the y-axis is true positive rate.

    Args:
    * `roc_file_path`: is the path to save the picture, if file_path=None, just show in memory.


Example:

    >>>from jddb.performance import Result, Report
    >>>result_csv_path = 'base/directory/to/save/test_result.csv'
    >>>report_csv_path = 'base/directory/to/save/test_report.csv'
    >>>test_result = Result(result_csv_path)
    >>>test_result.lucky_guess_threshold = .3 
    >>>test_result.tardy_alarm_threshold = .005
    >>>test_report = Report(report_csv_path)
    >>>test_report.add(test_result, "test_result")
    >>>test_report.remove(["test_result"])
    >>>test_report.plot_roc()
    >>>test_report.save()

    
 
