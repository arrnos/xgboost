# encoding:utf-8
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import os
from sklearn.cross_validation import train_test_split
from datapred import *
from model import xgb_model
import xgboost as xgb
from sklearn import metrics

file = feature_pred(data_read(filename='data/bank-full.csv'))
train_data, valid_data = train_test_split(file, test_size=0.3, random_state=1)
test_data = feature_pred(data_read(filename='data/bank.csv'))

# mode1. 训练model并保存
xgb_model(train_data, valid_data)

# mode2.调用保存好的model，进行预测
xgb_model = xgb.Booster(model_file='model/xgb.model')
xgb_test = xgb.DMatrix(test_data.iloc[:, :-1], test_data.iloc[:, -1])

y_pred = xgb_model.predict(xgb_test)
y_true = test_data.iloc[:, -1]

print zip(y_true,y_pred)
# 效果评价
classify_report = metrics.classification_report(y_true, y_pred)
confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
overall_accuracy = metrics.accuracy_score(y_true, y_pred)
acc_for_each_class = metrics.precision_score(y_true, y_pred, average=None)
average_accuracy = np.mean(acc_for_each_class)
score = metrics.accuracy_score(y_true, y_pred)
print '*********************** classify_report *****************************'
print classify_report
print '*********************** confusion_matrix ****************************'
print confusion_matrix
print '*********************** acc_for_each_class **************************'
print acc_for_each_class
print '*********************** average_accuracy: ***************************'
print('{0:f}'.format(average_accuracy))
print '*********************** overall_accuracy ****************************'
print('{0:f}'.format(overall_accuracy))
print '***********************    score   **********************************'
print('{0:f}'.format(score))
