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
import tensorflow as tf

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
train_data = new_data




reference_train_data,reference_train_label = handle_data.transform_data_to_compare_data(train_data, label, group_index_list, mirror_type, positive_value, negative_value)


reference_transformed_input_size = reference_train_data.shape[1]
single_input_size = reference_transformed_input_size / 2
reference_num_class = 1

x_reference = tf.placeholder(tf.float32, [None, reference_transformed_input_size])
y_true_reference = tf.placeholder(tf.float32, [None, reference_num_class])

hidden1 = tf.layers.dense(inputs=x_reference, units=2*reference_transformed_input_size, use_bias=True, activation=tf.nn.relu)
hidden2 = tf.layers.dense(inputs=hidden1, units=2*reference_transformed_input_size, use_bias=True, activation=tf.nn.relu)
hidden3 = tf.layers.dense(inputs=hidden2, units=reference_transformed_input_size, use_bias=True, activation=tf.nn.relu)
hidden4 = tf.layers.dense(inputs=hidden3, units=single_input_size, use_bias=True, activation=tf.nn.relu)
hidden5 = tf.layers.dense(inputs=hidden4, units=2*reference_num_class, use_bias=True, activation=tf.nn.relu)
# y_pred = tf.layers.dense(inputs=hidden1, units=4, activation=tf.nn.sigmoid)
y_pred = tf.layers.dense(inputs=hidden4, units=reference_num_class, activation=tf.nn.sigmoid)

reference_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true_reference, logits=y_pred_reference)

reference_cost = tf.reduce_mean(reference_loss)
optimizer = tf.train.AdamOptimizer(learning_rate=0.0005).minimize(reference_cost)



tf.add_to_collection('x_reference', x_reference)
tf.add_to_collection('y_true_reference', y_true_reference)
tf.add_to_collection('y_pred', y_pred)
tf.add_to_collection('reference_cost', reference_cost)
tf.add_to_collection('optimizer', optimizer)
saver = tf.train.Saver()



sess = tf.Session()
sess.run(tf.global_variables_initializer())

feed_dict_train = {
    x_reference       : reference_train_data,
    y_true_reference  : reference_train_label
}

for i in range(2500):
    reference_cost_val, reference_true_label, reference_pred_label, opt_obj = sess.run( [reference_cost, y_true_reference, y_pred_reference, optimizer], feed_dict=feed_dict_train )
    if (i % 100) == 0 :
        print('epoch: {0} cost = {1}'.format(i,reference_cost_val))


finish = clock()
saver.save(sess, model_name)

running_time = finish-start
print('running time is {0}'.format(running_time))











