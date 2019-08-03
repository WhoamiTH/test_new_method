import sys
sys.path.append('..')
import sklearn.svm as sksvm
import sklearn.linear_model as sklin
import sklearn.tree as sktree
from sklearn.externals import joblib
from time import clock
import handle_data
import predict_test
import pandas as pd
import numpy as np

def train_model(train_data, train_label):
    start = clock()
    global model_type
    if model_type == 'LR':
        model = sklin.LogisticRegression()
        model.fit(train_data,train_label.flatten())
    if model_type == 'SVC':
        model = sksvm.SVC(C=0.1,kernel='linear')
        # model = sksvm.SVC(C=0.1,kernel='rbf')
        # model = sksvm.SVC(C=0.1,kernel='poly')
        model.fit(train_data, train_label.flatten())
    if model_type == 'DT':
        model = sktree.DecisionTreeClassifier()
        model.fit(train_data, train_label.flatten())
    finish = clock()
    return model, finish-start


def set_para():
    global file_name
    global model_record_path
    global file_record_path
    global method_name
    global model_type
    global mirror_type
    global kernelpca_or_not
    global pca_or_not
    global num_of_components

    global scaler_name
    global pca_name
    global kernelpca_name
    global model_name

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
        if para[0] == 'model_type':
            model_type = para[1].upper()
        if para[0] == 'mirror_type':
            mirror_type = para[1]
        if para[0] == 'kernelpca':
            if para[1] == 'True':
                kernelpca_or_not = True
            else:
                kernelpca_or_not = False
        if para[0] == 'pca':
            if para[1] == 'True':
                pca_or_not = True
            else:
                pca_or_not = False
        if para[0] == 'num_of_components':
            num_of_components = int(para[1])

        if para[0] == 'scaler_name':
            scaler_name = para[1]
        if para[0] == 'pca_name':
            pca_name = para[1]
        if para[0] == 'kernelpca_name':
            kernelpca_name = para[1]
        if para[0] == 'model_name':
            model_name = para[1]

    if kernelpca_or_not and pca_or_not:
        pca_or_not = True
        kernelpca_or_not = False





# -------------------------------------parameters----------------------------------------
file_name = 'GData_train.csv'
model_record_path = '../1_year_result/model/'
file_record_path = '../1_year_result/record/'
method_name = "smote"
# model_type = 'LR'
model_type = 'SVC'
# model_type = 'DT'
# mirror_type = "mirror"
mirror_type = "not_mirror"
kernelpca_or_not = False
pca_or_not = False
num_of_components = 19

scaler_name = 'scaler.m'
pca_name = 'pca.m'
kernelpca_name = ''
model_name = 'model.m'
positive_value = 1
negative_value = -1
threshold_value = 0
winner_number = 3

# ----------------------------------set parameters---------------------------------------
set_para()

# ----------------------------------start processing-------------------------------------
print(file_name)

# file_number = re.findall(r"\d+", file_name)[-1]
scaler_name = model_record_path + method_name + '_' + scaler_name
if pca_or_not:
    pca_name = model_record_path + method_name + '_' + pca_name
if kernelpca_or_not:
    kernelpca_name = model_record_path  + method_name + '_' + kernelpca_name
model_name = model_record_path + method_name + '_' + model_name

data, label = handle_data.loadTrainData(file_name)

group_index_list = handle_data.group(data)

data = data.values
data = data.astype(np.float64)


start = clock()
new_data = handle_data.standarize_PCA_data(data, pca_or_not, kernelpca_or_not, num_of_components, scaler_name, pca_name, kernelpca_name)
train_data,train_label= handle_data.transform_data_to_compare_data(new_data, label, group_index_list, mirror_type, positive_value, negative_value)
model,training_time = train_model(train_data, train_label)
finish = clock()
joblib.dump(model, model_name)

running_time = finish-start
print(model)
print('running time is {0}'.format(running_time))