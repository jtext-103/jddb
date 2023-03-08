# How to use Performance

all time used here using ms as unit

## result 类

### properties

- tardy_alarm_threshold
- lucy_guess_threshold

### methods

- 构造函数，给一个 csv 文件路径

init that file with headers:

shot_no, predited_disruption, predicted_disruption_time, true_disrutpion, true_disruption_time, warning_time(这个如果是非破裂或者是预测错了的就写-1)

- add(shot, predicted_disruptoin, predicted_time)

- get_truth(meta_db)
  file the result csv with metadb result

- ture_positive()
- ture_positive_rate()
  还有其他的 metrics 自己加哈

- warning_time_histogram(file)
- accumulate_plot_warning_time_plot(file)
- average_warning_time()
- medean_warning_time()

## Roc Class

- 构造函数
  Roc(result_file_list)

- auc()

- plot(file)
