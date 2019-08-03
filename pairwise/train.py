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
    global reference_model_name

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
        if para[0] == 'reference_model_name':
            reference_model_name = para[1]

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

reference_model_name = 'reference_model'

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
reference_model_name = model_record_path + method_name + '_' + reference_model_name

data, label = handle_data.loadTrainData(file_name)

group_index_list = handle_data.group(data)

data = data.values
data = data.astype(np.float64)


start = clock()
new_data = handle_data.standarize_PCA_data(data, pca_or_not, kernelpca_or_not, num_of_components, scaler_name, pca_name, kernelpca_name)
train_data = new_data


single_input_size = train_data.shape[1]
# transformed_input_size = 2 * single_input_size


num_class = 1

# x = tf.placeholder(tf.float32, [None, single_input_size])
x = tf.placeholder(tf.float32, [None, 2, single_input_size])
y_true = tf.placeholder(tf.float32, [None, 2, num_class])
y_transformed_pred = tf.placeholder(tf.float32, [None, num_class])
y_transformed_true = tf.placeholder(tf.float32, [None, num_class])


# one hidden layer ------------------------------------------------
hidden1 = tf.layers.dense(inputs=x, units=2*single_input_size, use_bias=True, activation=tf.nn.relu)
# y_pred = tf.layers.dense(inputs=hidden1, units=4, activation=tf.nn.sigmoid)
y_pred = tf.layers.dense(inputs=hidden1, units=num_class, activation=tf.nn.sigmoid)


y_transformed = tf.math.sigmoid(10 * (y_pred[:,0,0] -  y_pred[:,1,0]))
y_transformed = tf.reshape(y_transformed, shape=(-1,1))

loss_1 = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)
loss_2 = tf.square((y_transformed - y_transformed_true) * tf.math.log(y_transformed_pred / (1- y_transformed)))

cost = tf.reduce_mean(loss)
optimizer = tf.train.AdamOptimizer(learning_rate=0.0005).minimize(cost)




tf.add_to_collection('x', x)
tf.add_to_collection('y_true', y_true)
tf.add_to_collection('y_pred', y_pred)
tf.add_to_collection('cost', cost)
tf.add_to_collection('optimizer', optimizer)
saver = tf.train.Saver()


# create session and run the model
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.46)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
sess.run(tf.global_variables_initializer())
for i in range(train_times):
    # train_data, train_label = handle_data.generate_batch_data(positive_data, negative_data, batch_size)

    # train_data, train_label, transformed_label = handle_data.next_batch(positive_data, negative_data)
    current_train_data, current_train_label, current_transformed_label, current_transformed_pred = handle_data.next_batch(train_data, train_label, group_index_list, reference_model_name=reference_model_name)

    current_train_data = np.array(current_train_data).reshape((-1,2,single_input_size))
    current_train_label = np.array(current_train_label).reshape((-1,2,1))
    current_transformed_label = np.array(current_transformed_label).reshape((-1,1))
    current_transformed_pred = np.array(current_transformed_pred).reshape((-1,1))
    feed_dict_train = {
        x                   : current_train_data,
        y_true              : current_train_label,
        y_transformed_true  : current_transformed_label,
        y_transformed_pred  : current_transformed_pred

    }

    cost_val, true_label, pred_label, opt_obj = sess.run( [cost, y_true, y_pred,
        optimizer], feed_dict=feed_dict_train )
    if (i % 1000) == 0 :
        print('epoch: {0} cost = {1}'.format(i,cost_val))
#             print(pred_label)
#             print(true_label)
# print(pred_label)


finish = clock()

saver.save(sess, model_name)
running_time = finish-start
print('running time is {0}'.format(running_time))











