# encoding:utf-8
import numpy as np
import pandas as pd
import xgboost as xgb
import os


def xgb_model(train_data, valid_data):
    # XGBoost参数设置
    params = {
        'booster': 'gbtree',
        'objective': 'multi:softmax',  # 多分类的问题
        'num_class': 2,  # 类别数，与 multisoftmax 并用
        'gamma': 0.2,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
        'max_depth': 11,  # 构建树的深度，越大越容易过拟合
        'lambda': 3,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
        'subsample': 0.7,  # 随机采样训练样本
        'colsample_bytree': 0.7,  # 生成树时进行的列采样
        'min_child_weight': 0.1,
        'silent': 0,  # 设置成1则没有运行信息输出，最好是设置为0.
        # 'eta': 0.007,  # 如同学习率
        'eta': 0.3,  # 如同学习率
        'seed': 1000,
        # 'eval_metric': 'auc'
    }


    # xgb Dmatrix 初始化
    xgb_train = xgb.DMatrix(train_data.iloc[:, :-1], train_data.iloc[:, -1])
    xgb_valid = xgb.DMatrix(valid_data.iloc[:, :-1], valid_data.iloc[:, -1])

    num_rounds = 1000  # 迭代次数
    watchlist = [(xgb_train, 'train'), (xgb_valid, 'valid')]

    # 训练模型并保存
    # early_stopping_rounds 当设置的迭代次数较大时，early_stopping_rounds 可在一定的迭代次数内准确率没有提升就停止训练
    model = xgb.train(params, xgb_train, num_rounds, watchlist, early_stopping_rounds=100)
    model.save_model('model/xgb.model')  # 用于存储训练出的模型
    print "best best_ntree_limit", model.best_ntree_limit
