# encoding:utf-8
import numpy as np
import pandas as pd
import os
from sklearn.datasets import dump_svmlight_file

os.chdir('/Users/arrnos/PycharmProjects/xgboost')


def data_read(filename='data/bank.csv'):
    file = pd.read_csv(filename, sep=';')
    return file


def feature_pred(file):
    selected_col = ['age', 'job', 'marital', 'education', 'balance', 'day', 'month',
                    'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']

    job_map = {'management': 0,
               'retired': 1,
               'self-employed': 2,
               'unknown': 3,
               'unemployed': 4,
               'housemaid': 5,
               'admin.': 6,
               'technician': 7,
               'student': 8,
               'services': 9,
               'entrepreneur': 10,
               'blue-collar': 11}

    marital_map = {'single': 0,
                   'married': 1,
                   'divorced': 2}

    education_map = {'unknown': 0,
                     'primary': 1,
                     'tertiary': 2,
                     'secondary': 3}

    month_map = {'mar': 0,
                 'feb': 1,
                 'aug': 2,
                 'sep': 3,
                 'may': 4,
                 'jun': 5,
                 'jul': 6,
                 'jan': 7,
                 'apr': 8,
                 'nov': 9,
                 'dec': 10,
                 'oct': 11}
    poutcome_map = {'unknown': 0,
                    'other': 1,
                    'success': 2,
                    'failure': 3}

    y_map = {'yes': 1,
             'no': 0}

    file['job'] = file['job'].map(job_map)
    file['marital'] = file['marital'].map(marital_map)
    file['education'] = file['education'].map(education_map)
    file['month'] = file['month'].map(month_map)
    file['poutcome'] = file['poutcome'].map(poutcome_map)
    file['y'] = file['y'].map(y_map)

    return file[selected_col]


if __name__ == '__main__':
    file = data_read()
    feature_pred(file)