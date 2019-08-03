import sys
sys.path.append('..')
from sklearn.externals import joblib
from time import clock
import handle_data
import predict_test
import numpy as np
import csv
import re
import pandas as pd


def set_para():
    global file_name
    global model_record_path
    global file_record_path
    global method_name

    global scaler_name
    global kernelpca_name
    global pca_name
    global model_name
    global record_name

    argv = sys.argv[1:]
    for each in argv:
        para = each.split('=')
        if para[0] == 'file_name':
            file_name = para[1]
        if para[0] == 'model_record_path':
            model_record_path = para[1]
        if para[0] == 'file_record_path':
            file_record_path = para[1]
        if para[0] == 'method_name':
            method_name = para[1]

        if para[0] == 'scaler_name':
            scaler_name = para[1]
        if para[0] == 'kernelpca_name':
            kernelpca_name = para[1]
        if para[0] == 'pca_name':
            pca_name = para[1]
        if para[0] == 'model_name':
            model_name = para[1]
        if para[0] == 'record_name':
            record_name = para[1]

    if kernelpca_name and pca_name:
        kernelpca_name = ''




# -------------------------------------global parameters-------------------------------
file_name = 'NBA_test.csv'
model_record_path = '../1_year_result/model/'
file_record_path = '../1_year_result/record/'
method_name = "smote"

scaler_name = 'scaler.m'
kernelpca_name = ''
pca_name = ''
model_name = 'model.m'
threshold_value = 0
record_name = 'result.csv'
# ----------------------------------set parameters--------------------------------------
set_para()

# ----------------------------------start processing------------------------------------
print(file_name)

scaler_name = model_record_path + method_name + '_' + scaler_name
if pca_name != '':
    pca_name = model_record_path + method_name + '_' + pca_name
if kernelpca_name != '':
    kernelpca_name = model_record_path  + method_name + '_' + kernelpca_name
model_name = model_record_path  + method_name + '_' + model_name
record_name = file_record_path + method_name + '_' + record_name

print(model_name)

file_data, data = handle_data.loadTestData(file_name)

group_index_list = handle_data.group(data)

data = data.values
data = data.astype(np.float64)
model = joblib.load(model_name)
test_group_num = len(group_index_list)

all_result = []
start = clock()
new_data = handle_data.transform_data_by_standarize_pca(data, scaler_name, pca_name, kernelpca_name)
for num in range(test_group_num):
    print('the {0} group race'.format(num+1))
    current_group_data = new_data[group_index_list[num]]
    all_result += predict_test.group_test(current_group_data, model, threshold_value)
finish = clock()

file_data['predict_result'] = all_result
file_data.to_csv(record_name, index=False)
print('Done')