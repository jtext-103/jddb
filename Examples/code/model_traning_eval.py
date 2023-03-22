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
from jddb.performance import Result
from jddb.performance import Report
from jddb.performance import Roc
from jddb.file_repo import FileRepo


def matrix_build(shot_list, file_repo, tag_list_matrix):
    X_set = []
    y_set = []
    for i in range(len(shot_list)):
        x_data = file_repo.read_data(shot_list[i], tag_list_matrix)
        res = np.array([list(x_data.values())]).T
        X_set.append(np.delete(res, 3, 1))
        y_set = res[:, 3]
    return X_set, y_set


def shot_result(y_red, threshold):
    binary_result = 1 * (y_pred >= threshold)
    for k in range(len(binary_result) - 2):
        if binary_result[k] + binary_result[k + 1] + binary_result[k + 2] == 3:
            predicted_disruption_time = (k + 2) / 1000
            predicted_disruption = 1
            break
        else:
            predicted_disruption_time = -1
            predicted_disruption = 0
    return predicted_disruption, predicted_disruption_time


if __name__ == '__main__':
    # init FileRepo and MetaDB
    # %%
    test_file_repo = FileRepo(os.path.join(r'H:\rt\jddb\jddb\Examples\FileRepo', "$shot_2$00", "$shot_1$0"))
    test_shot_list = test_file_repo.get_all_shots()
    tag_list = test_file_repo.get_tag_list(test_shot_list[0])
    y = []
    for i in test_shot_list:
        dis_label = test_file_repo.read_labels(i, ['IsDisrupt'])
        y.append(dis_label['IsDisrupt'])

    # train test split on shot not sample
    # %%
    train_list, val_list, train_y, val_y = \
        train_test_split(test_shot_list, y, test_size=0.2, random_state=1, shuffle=True, stratify=y)

    # # create x and y matrix for ML models
    # # %%
    X_train, y_train = matrix_build(train_list, test_file_repo, tag_list)
    X_val, y_val = matrix_build(val_list, test_file_repo, tag_list)
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
    performance_test = Result(r'..\test\test_result.xlsx')
    sample_result = dict()
    for i in range(len(val_list)):
        X, y_val = matrix_build(val_list[i], test_file_repo, tag_list)
        y_pred = gbm.predict(X, num_iteration=gbm.best_iteration)
        sample_result.setdefault(val_list[i], []).append(y_pred)

        # using the sample reulst to predict disruption on shot, and save result to result file using result module.
        predicted_disruption, predicted_disruption_time = shot_result(y_pred, 0.5)
        shot_no = val_list[i]
        true_disruption = val_y[i]
        if true_disruption:
            true_disruption_time = test_file_repo.read_labels(val_list[i], ['DownTime'])
        else:
            true_disruption_time = -1
        performance_test.result.loc[
            i, ['shot_no', 'predicted_disruption', 'predicted_disruption_time', 'true_disruption',
                'true_disruption_time']] = [shot_no, predicted_disruption, predicted_disruption_time,
                                            true_disruption, true_disruption_time]
    performance_test.lucky_guess_threshold = .1
    performance_test.tardy_alarm_threshold = .005
    performance_test.calc_metrics()
    # print(performance_test.get_precision())
    # performance_test.warning_time_histogram('../test/', 5)
    # performance_test.accumulate_warning_time_plot('../test/', tp+fp)
    performance_test.save()

    # simply chage different disruptivity triggering level and logic, get many result.
    test_report = Report('../test/report.xlsx')
    test_report.report_file()
    level = np.linspace(0, 1, 50)
    for i in level:
        # %% evaluation
        for j in val_y:
            y_pred = sample_result[j][0]
            predicted_disruption, predicted_disruption_time = shot_result(y_pred, i)
            performance_test.result.loc[
                performance_test.result[performance_test.result.shot_no == j].index, ['predicted_disruption',
                                                                                      'predicted_disruption_time']] = [
                predicted_disruption, predicted_disruption_time]

        # add result to the report
        test_report.add(performance_test, str(i), .005, .1)

    # find best result

    # plot all metrics
    test_roc = Roc('../test/report.xlsx')
    test_roc.roc('../test/')
