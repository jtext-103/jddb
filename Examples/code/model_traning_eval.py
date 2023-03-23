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


def matrix_build(shot_list, file_repo, tags):
    """
    get x and y from file_repo with shots and tags
    Args:
        shot_list: shots for data matrix
        file_repo:
        tags: tags from file_repo

    Returns: matrix of x and y

    """
    x_set = []
    y_set = []
    for i in range(len(shot_list)):
        x_data = file_repo.read_data(shot_list[i], tags)
        res = np.array([list(x_data.values())]).T
        x_set.append(np.delete(res, 3, 1))
        y_set = res[:, 3]
    return x_set, y_set


def get_shot_result(y_red, threshold_sample):
    """
    get shot result with a threshold
    Args:
        y_red: sample result from model
        threshold_sample: disruptive predict level

    Returns: shot predict result and predict time

    """
    binary_result = 1 * (y_pred >= threshold_sample)
    for k in range(len(binary_result) - 2):
        if binary_result[k] + binary_result[k + 1] + binary_result[k + 2] == 3:
            predicted_dis_time = (k + 2) / 1000
            predicted_dis = 1
            break
        else:
            predicted_dis_time = -1
            predicted_dis = 0
    return predicted_dis, predicted_dis_time


if __name__ == '__main__':
    # init FileRepo and MetaDB
    # %%
    test_file_repo = FileRepo(os.path.join(r'H:\rt\jddb\jddb\Examples\FileRepo', "$shot_2$00", "$shot_1$0"))
    test_shot_list = test_file_repo.get_all_shots()
    tag_list = test_file_repo.get_tag_list(test_shot_list[0])
    is_disruot = []
    for threshold in test_shot_list:
        dis_label = test_file_repo.read_labels(threshold, ['IsDisrupt'])
        is_disruot.append(dis_label['IsDisrupt'])

    # train test split on shot not sample
    # %%
    train_shots, test_shots, _, _ = \
        train_test_split(test_shot_list, is_disruot, test_size=0.2, random_state=1, shuffle=True, stratify=is_disruot)

    # # create x and y matrix for ML models
    # # %%
    X_train, y_train = matrix_build(train_shots, test_file_repo, tag_list)
    X_test, y_test = matrix_build(test_shots, test_file_repo, tag_list)
    lgb_train = lgb.Dataset(X_train, y_train)  # create dataset for LightGBM

    # use LightGBM to train a model.
    # %%
    #hyper-parameters
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
                    valid_sets={lgb_train},
                    evals_result=evals_result,
                    early_stopping_rounds=30)

    # save sample result to a dict, so when predicting shot with differnet trgging logic,
    # you don't have to re-infor the testshot
    # 改一下注释
    test_result = Result(r'..\test\test_result.csv')
    sample_result = dict()
    shot_nos=test_shots
    shots_pred_disrurption=[]
    shots_pred_disruption_time=[]
    for shot in test_shots:
        X, _ = matrix_build([shot], test_file_repo, tag_list)
        y_pred = gbm.predict(X, num_iteration=gbm.best_iteration)
        sample_result.setdefault(shot, []).append(y_pred)  # save sample result to a dict

    # using the sample reulst to predict disruption on shot, and save result to result file using result module.
        predicted_disruption, predicted_disruption_time = get_shot_result(y_pred, .5)
        shots_pred_disrurption.append(predicted_disruption)
        shots_pred_disruption_time.append(predicted_disruption_time)
    test_result.add(shot_nos, shots_pred_disrurption, shots_pred_disruption_time)
    test_result.get_all_truth(shot_nos)
    test_result.lucky_guess_threshold = .1
    test_result.tardy_alarm_threshold = .005
    test_result.calc_metrics()
    test_result.save()
    print("precision = " + str(test_result.get_precision()))
    print("tpr = " + str(test_result.ture_positive_rate()))
    test_result.confusion_matrix()
    test_result.warning_time_histogram('../test/', [-1,.002,.01,.05,.1,.3])
    test_result.accumulate_warning_time_plot('../test/')

    # simply chage different disruptivity triggering level and logic, get many result.
    test_report = Report('../test/report.csv')
    thresholds = np.linspace(0, 1, 50)
    for threshold in thresholds:
        # %% evaluation
        shot_nos = test_shots
        shots_pred_disrurption = []
        shots_pred_disruption_time = []
        for shot in test_shots:
            y_pred = sample_result[shot][0]
            predicted_disruption, predicted_disruption_time = get_shot_result(y_pred, threshold)
            shots_pred_disrurption.append(predicted_disruption)
            shots_pred_disruption_time.append(predicted_disruption_time)
        # i dont save so the file never get created
        temp_test_result=Result('../test/temp_result.csv')
        temp_test_result.lucky_guess_threshold = .1
        temp_test_result.tardy_alarm_threshold = .005
        temp_test_result.add(shot_nos, shots_pred_disrurption, shots_pred_disruption_time)
        temp_test_result.get_all_truth(shot_nos)

        # add result to the report
        test_report.add(temp_test_result, "thr="+str(threshold))

    # find best result

    # plot all metrics
    test_report.plot_roc(../test/)
