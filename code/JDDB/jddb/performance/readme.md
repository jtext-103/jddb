# How to use Performance （小玉）

all time used here, using s as unit

warning time is positive

## result 类

### properties

- tardy_alarm_threshold
- lucky_guess_threshold

### methods

- 构造函数，给一个 csv 文件路径

init that file with headers:

shot_no, predited_disruption, predicted_disruption_time, true_disrutpion, true_disruption_time, warning_time,true_positive,false_positive(这个如果是非破裂或者是预测错了的就写-1)

- add(shot, predicted_disruptoin, predicted_time)

- get_all_truth(meta_db)
  fill the result csv with metadb result, it also calls recalc_metric

- recalc_metrics()

- ture_positive()
- ture_positive_rate()
  还有其他的 metrics 自己加哈

- warning_time_histogram(file)
- accumulate_warning_time_plot(file)
- average_warning_time()
- medean_warning_time()

## report class

- report(report_file)
- add(result,model_name)

## Roc Class

- 构造函数
  Roc(result_file_list)

- auc()

- plot(file)
