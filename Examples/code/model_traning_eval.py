# %%
# this examlpe shows how to build a ML mode to predict disruption and
# evaluate its performance using jddb
# this depands on the output FileRepo of basic_data_processing.py

# %%
import numpy as np
import lightgbm as lgb
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from performance.result import Result
from performance.report import Report
from performance.roc import Roc

if __name__ == '__main__':
    # init FileRepo and MetaDB
    # %%
    df_train = pd.read_csv('../test/topdata_train.csv')
    df_val = pd.read_csv('../test/topdata_val.csv')
    shot_info = np.load('../test/L_beta_val.npy')

    # train test split on shot not sample

    # %%

    # create x and y matrix for ML models
    # %%
    y_train = df_train['disrup_tag']
    X_train = df_train.drop(['disrup_tag', '#', 'time', 'endtime'], axis=1).values
    y_val = df_val['disrup_tag']
    X_val = df_val.drop(['disrup_tag', '#', 'time', 'endtime'], axis=1).values
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train)

    # use LightGBM to train a model.
    # %%
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'auc'},
        'max_depth': 9,
        'num_leaves': 70,
        'learning_rate': 0.1,
        'feature_fraction': 0.86,
        'bagging_fraction': 0.73,
        'bagging_freq': 0,
        'verbose': 0,
        'cat_smooth': 10,
        'max_bin': 255,
        'min_data_in_leaf': 165,
        'lambda_l1': 0.03,
        'lambda_l2': 2.78,
        'is_unbalance': True,
        'min_split_gain': 0.3

    }
    evals_result = {}  # to record eval results for plotting
    print('Starting training...')
    # train
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=300,
                    valid_sets={lgb_train, lgb_eval},
                    evals_result=evals_result,
                    early_stopping_rounds=30)

    # infer on test shots

    inf_result = gbm.predict(X_val, num_iteration=gbm.best_iteration)

    # save sample result to a FileRepo, so when predicting shot with differnet trgging logic,
    # you don't have to re-infor the testshot
    df_result = pd.DataFrame(columns=['shot_no', 'predicted_disruption', 'predicted_disruption_time', 'true_disruption',
                                      'true_disruption_time', 'warning_time', 'true_positive', 'false_positive',
                                      'tardy_alarm_threshold', 'lucky_guess_threshold'])
    for i in range(shot_info.shape[0]):
        va_shot = shot_info[i, 0]
        if shot_info[i, 1]:
            shot_info[i, 1] = 1
        dis = df_val[df_val['#'] == va_shot]
        X = dis.drop(['disrup_tag', '#', 'time', 'endtime'], axis=1).values
        y_pred = gbm.predict(X, num_iteration=gbm.best_iteration)
        binary_result = 1 * (y_pred >= 0.5)
        for k in range(len(binary_result) - 2):
            if binary_result[k] + binary_result[k + 1] + binary_result[k + 2] == 3:
                predicted_disruption_time = (k + 2) / 1000
                predicted_disruption = 1
                break
            else:
                predicted_disruption_time = -1
                predicted_disruption = 0
        shot_no = va_shot
        true_disruption = shot_info[i, 1]
        if true_disruption:
            true_disruption_time = shot_info[i, 2] / 1000
        else:
            true_disruption_time = -1
        index = len(df_result)
        df_result.loc[index, ['shot_no', 'predicted_disruption', 'predicted_disruption_time', 'true_disruption',
                              'true_disruption_time']] = [shot_no, predicted_disruption, predicted_disruption_time,
                                                          true_disruption, true_disruption_time]

    df_result.to_excel(r'..\test\test_result.xlsx', index=False)

    # using the sample reulst to predict disruption on shot, and save result to result file using result module.
    performance_test = Result(r'..\test\test_result.xlsx')
    performance_test.lucky_guess_threshold=.1
    performance_test.tardy_alarm_threshold=.005
    performance_test.calc_metrics()
    print(performance_test.get_precision())
    performance_test.warning_time_histogram('../test/',5)
    # performance_test.accumulate_warning_time_plot('../test/', tp+fp)
    performance_test.save()

    # simply chage different disruptivity triggering level and logic, get many result.
    test_report = Report('../test/report.xlsx')
    test_report.report_file()
    for i in range(10,200,10):

    # %% evaluation

    # add result to the report
        test_report.add(performance_test, str(i), .005, i*.001)

    # find best result

    # plot all metrics
    test_roc = Roc('../test/report.xlsx')
    test_roc.roc('../test/')
