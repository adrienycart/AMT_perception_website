import codecs
import os
import random
import cPickle as pickle

import numpy as np
import tensorflow as tf

import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
import pandas as pd

from config import MAX_ANSWERS


BATCH_SIZE = 50




def get_y_z_alphas(ratings):

    y = (ratings==int(round(MAX_ANSWERS/2))).astype(int)
    z = (ratings>int(round(MAX_ANSWERS/2))).astype(int)
    # alpha is between 0 and 0.5
    alphas = np.abs(ratings-int(round(MAX_ANSWERS/2)))/int(round(MAX_ANSWERS))

    return y,z,alphas

def linear_regression_model(features,weights):
    #features_o and features_t are of shape: [batch_size, n_features]

    assert features.get_shape().as_list()[1] == N_FEATURES
    output = tf.sigmoid(tf.matmul(features, weights))

    return output


def contrastive_loss(batch1,batch2,y,alphas):
    # y[i] = 1 iff batch1[i] and batch2[i] were rated equally similar

    loss = y * tf.square(batch1 - batch2) + \
           (1-y)*tf.square(tf.maximum(alphas-tf.abs(batch1-batch2),0))
    return tf.reduce_mean(loss)

def contrastive_loss_magnitude(batch1,batch2,y,z,alphas):
    # y[i] = 1 iff batch1[i] and batch2[i] were rated equally
    # z[i] = 1 iff batch1[i] was better rated than batch2[i]

    loss = y * tf.square(batch1 - batch2) + \
           (1-y)*tf.square(tf.maximum(alphas-tf.abs(batch1-batch2),0)) + \
           (1-y)*(z*tf.square(batch2) + (1-z)*tf.square(batch1))
    return tf.reduce_mean(loss)

def contrastive_loss_absolute(batch1,batch2,y,z,alphas):
    # y[i] = 1 iff batch1[i] and batch2[i] were rated equally
    # z[i] = 1 iff batch1[i] was better rated than batch2[i]

    loss = y * tf.square(batch1 - batch2) + \
           (1-y)*tf.square(tf.maximum(alphas-z*(batch1-batch2)-(1-z)*(batch2-batch1),0))
    return tf.reduce_mean(loss)

def import_features(example_dir,system,features_to_use):
    results = pickle.load(open(os.path.join(example_dir,system+'.pkl'), "rb"))
    all_feat = []
    for feat in features_to_use:
        value = results[feat]
        if type(value) is tuple:
            all_feat += list(value)
        elif type(value) is float:
            all_feat += [np.float64(value)]
        else:
            all_feat += [value]

    all_feat = [float(elt) for elt in all_feat]

    return all_feat

def shuffle(*args):
    assert all([arg.shape[0] == args[0].shape[0] for arg in args])
    n = args[0].shape[0]
    shuffle_idx = list(range(n))
    random.shuffle(shuffle_idx)
    output = []
    for arg in args:
        output += [arg[shuffle_idx]]
    return output

def sample(n_samples,*args):
    assert all([arg.shape[0] == args[0].shape[0] for arg in args])
    n = args[0].shape[0]
    sample_idx = list(range(n))
    sample_idx = random.sample(sample_idx, n_samples)
    output = []
    for arg in args:
        output += [arg[sample_idx]]
    return output

#### Prepare data:
feature_dir = 'precomputed_features'

filecp = codecs.open('db_csv/answers_data.csv', encoding = 'utf-8')
answers = np.genfromtxt(filecp,dtype=object,delimiter=";")
answers = answers[1:,:]



#################################################@
####### MIXED EFFECTS MODEL
#################################################@


AQ = pd.read_csv('db_csv/answers_data.csv',delimiter=';')

results_dict = {}
feature_dir = 'precomputed_features'

systems = ['cheng','google',"kelz","lisu"]
pairs = []
for i in range(len(systems)):
    for j in range(i+1,len(systems)):
        pairs += [[systems[i],systems[j]]]

r = np.array(list(range(len(pairs))))

for example in np.unique(answers[:,1]):
    example_dir = os.path.join(feature_dir,example)
    results_dict[example] = {}
    for system in systems:
        results = pickle.load(open(os.path.join(example_dir,system+'.pkl'), "rb"))
        results_dict[example][system]=results

f_syst1 = []
f_syst2 = []
f_diff = []
for row in answers:
    f_syst1 += [results_dict[row[1]][row[2]]["notewise_On_50"][-1]]
    f_syst2 += [results_dict[row[1]][row[3]]["notewise_On_50"][-1]]
    f_diff += [abs(results_dict[row[1]][row[2]]["notewise_On_50"][-1]-results_dict[row[1]][row[3]]["notewise_On_50"][-1])]

data = {'f_system1':f_syst1,'f_system2':f_syst2, 'f_diff':f_diff}
data_f_mes = pd.DataFrame(data)

filecp = codecs.open('db_csv/user_data.csv', encoding = 'utf-8')
users = np.genfromtxt(filecp,dtype=object,delimiter=";")
users = users[1:,:]

goldmsi = []
for row in answers:
    user_id = row[4]
    goldmsi += [float(users[users[:,0]==user_id,5])]
data = {'goldmsi':goldmsi}
data_goldmsi = pd.DataFrame(data)

AQ_new = pd.concat([AQ, data_f_mes,data_goldmsi],axis=1)
reverse_idx = AQ_new['f_system1']>AQ_new['f_system2']


####       0            1         2          3        4         5        6            7           8        9           10             11
#### ['question_id' 'example' 'system1' 'system2' 'user_id' 'answer' 'recognised' 'difficulty' 'time'  'f_system1' , 'f_system2', 'goldmsi']

# print AQ_new[['question_id','system1','system2','answer']]

system1 = AQ_new["system1"][reverse_idx]
f_system1 = AQ_new["f_system1"][reverse_idx]
AQ_new.loc[reverse_idx, 'system1']=AQ_new["system2"][reverse_idx]
AQ_new.loc[reverse_idx, 'system2']=system1
AQ_new.loc[reverse_idx, 'f_system1']=AQ_new["f_system2"][reverse_idx]
AQ_new.loc[reverse_idx, 'f_system2']=f_system1
AQ_new.loc[reverse_idx, 'answer']=1- AQ_new["answer"][reverse_idx]

# print AQ_new[['question_id','system1','system2','answer']]



### Co-dependent variables: *
### Random variables: +(var/user_id) --> /user_id
### Also check multiple regression (simple)
### Also check multiple logistic regression

### Mixed Effects Model
mixed = smf.mixedlm("answer ~ f_system1*f_system2+goldmsi+recognised+difficulty", AQ_new,groups='question_id')
mixed_fit = mixed.fit()
print(mixed_fit.summary())

### logistic regression
# feature_columns = ["recognised","difficulty","f_system1","f_system2","goldmsi"]
# X = AQ_new.loc[:, feature_columns].values
#
# X = np.concatenate([X,X[:,4:5]-X[:,3:4]],axis=1)
# print X.shape
# y=AQ_new.answer
# clf = LogisticRegression().fit(X, y)
# print clf.coef_
# for c,feat in zip(clf.coef_[0],feature_columns+['f_diff']):
#     print feat, 'coef', c


########################################################
###########      USING FEATURES
########################################################


# #### GATHER DATA
# features1=[]
# features2=[]
# ratings=[]
#
# features_to_use = [
#                 "framewise",
#                 "notewise_On_50",
#                 "notewise_OnOff_50_0.2",
#                 "high_f",
#                 "low_f",
#                 "high_n",
#                 "low_n",
#
#                 "loud_fn",
#                 "loud_ratio_fn",
#
#                 "out_key",
#                 "out_key_bin",
#
#                 "repeat",
#                 "merge",
#
#                 "semitone_f",
#                 "octave_f",
#                 "third_harmonic_f",
#                 "semitone_n",
#                 "octave_n",
#                 "third_harmonic_n",
#                 ]
#
# labels = [
#     "framewise_P",
#     "framewise_R",
#     "framewise_F",
#     "notewise_On_P",
#     "notewise_On_R",
#     "notewise_On_F",
#     "notewise_OnOff_P",
#     "notewise_OnOff_R",
#     "notewise_OnOff_F",
#
#     "high_f_P",
#     "high_f_R",
#     "high_f_F",
#     "low_f_P",
#     "low_f_R",
#     "low_f_F",
#     "high_n_P",
#     "high_n_R",
#     "high_n_F",
#     "low_n_P",
#     "low_n_R",
#     "low_n_F",
#
#     "loud_fn",
#     "loud_ratio_fn",
#
#     "out_key_fp",
#     "out_key_all",
#     "out_key_bin_fp",
#     "out_key_bin_all",
#
#     "repeat_fp",
#     "repeat_all",
#     "merge_fp",
#     "merge_all",
#
#     "semitone_f_fp",
#     "semitone_f_all",
#     "octave_f_fp",
#     "octave_f_all",
#     "third_harmonic_f_fp",
#     "third_harmonic_f_all",
#     "semitone_n_fp",
#     "semitone_n_all",
#     "octave_n_fp",
#     "octave_n_all",
#     "third_harmonic_n_fp",
#     "third_harmonic_n_all",
#     ]
#
# for q_id in np.unique(answers[:,0]):
#     data = answers[answers[:,0]==q_id]
#     vote = np.sum(data[:,5].astype(int))
#     example = data[0,1]
#     system1 = data[0,2]
#     system2 = data[0,3]
#
#     # print example, system1, system2
#
#     example_dir = os.path.join(feature_dir,example)
#     # print len(import_features(example_dir,system1)),import_features(example_dir,system1)
#
#     features1 += [import_features(example_dir,system1,features_to_use)]
#     features2 += [import_features(example_dir,system2,features_to_use)]
#     ratings += [vote]
#
# features1 = np.array(features1,dtype=float)
# features2 = np.array(features2,dtype=float)
# ratings = np.array(ratings,dtype=int)
#
# N_FEATURES = features1.shape[1]
#
#
# y,z,alpha = get_y_z_alphas(ratings)
# y = y[:,None]
# z = z[:,None]
# alpha = alpha[:,None]
#
# #### SPLIT DATA
#
# n_questions = len(ratings)
# n_train = int(0.8*n_questions)
# n_valid = int(0.1*n_questions)
#
# features1_train = features1[:n_train]
# features2_train = features2[:n_train]
# y_train = y[:n_train]
# z_train = z[:n_train]
# alpha_train = alpha[:n_train]
#
# features1_valid = features1[n_train:n_train+n_valid]
# features2_valid = features2[n_train:n_train+n_valid]
# y_valid = y[n_train:n_train+n_valid]
# z_valid = z[n_train:n_train+n_valid]
# alpha_valid = alpha[n_train:n_train+n_valid]
#
# features1_test = features1[n_train+n_valid:]
# features2_test = features2[n_train+n_valid:]
# y_test = y[n_train+n_valid:]
# z_test = z[n_train+n_valid:]
# alpha_test = alpha[n_train+n_valid:]
#
# features1_ph = tf.placeholder(tf.float32,[None,N_FEATURES])
# features2_ph = tf.placeholder(tf.float32,[None,N_FEATURES])
# y_ph = tf.placeholder(tf.float32,[None,1])
# z_ph = tf.placeholder(tf.float32,[None,1])
# alpha_ph = tf.placeholder(tf.float32,[None,1])
#
# weights = tf.Variable(tf.ones([N_FEATURES,1]),name='weights')
#
# model_output1 = linear_regression_model(features1_ph,weights)
# model_output2 = linear_regression_model(features2_ph,weights)
#
#
#
# loss = contrastive_loss_absolute(model_output1,model_output2,y_ph,z_ph,alpha_ph)
# optimize = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)
#
# valid_costs = []
#
# feed_dict_valid = {
#     features1_ph:features1_valid,
#     features2_ph:features2_valid,
#     y_ph:y_valid,
#     z_ph:z_valid,
#     alpha_ph: alpha_valid,
#     }
#
# feed_dict_test = {
#     features1_ph:features1_test,
#     features2_ph:features2_test,
#     }
#
# best_valid = None
# best_parameters = None
#
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
#
# for i in range(2000):
#
#     features1_batch ,features2_batch,y_batch ,z_batch ,alpha_batch = sample(BATCH_SIZE,features1,features2,y,z,alpha)
#     feed_dict_train = {
#         features1_ph:features1_batch,
#         features2_ph:features2_batch,
#         y_ph:y_batch,
#         z_ph:z_batch,
#         alpha_ph: alpha_batch,
#         }
#
#
#     sess.run(optimize, feed_dict=feed_dict_train)
#     valid_cost = sess.run(loss, feed_dict=feed_dict_valid)
#
#     #Compute agreement, removing draws:
#     non_draws = y_valid==0
#     z_non_draws = z_valid[non_draws]
#     metrics1,metrics2 = sess.run([model_output1,model_output2],feed_dict_valid)
#     result_metrics = (metrics1<metrics2).astype(int)
#     result_metrics = result_metrics[non_draws]
#
#     print i, valid_cost, np.mean((z_non_draws==result_metrics).astype(int))
#
#     if best_valid is None or valid_cost<best_valid:
#         best_parameters = sess.run(weights)
#
# ###### RESULTS
#
#
# print 'Best parameters:'
# for (label,value) in zip(labels,best_parameters):
#     print label, value
#
#
#
# ## Remove draws:
# non_draws = y_test==0
# z_non_draws = z_test[non_draws]
# metrics1,metrics2 = sess.run([model_output1,model_output2],feed_dict_test)
# for m1,m2 in zip(metrics1,metrics2):
#     print m1,m2
# result_metrics = (metrics1<metrics2).astype(int)
# result_metrics = result_metrics[non_draws]
# results_F1 = (features1_test[:,5:6] <features2_test[:,5:6]).astype(int)
# results_F1 = results_F1[non_draws]
#
# print "average agreement new metric:", np.mean((z_non_draws==result_metrics).astype(int))
# print "average agreement F-measure:", np.mean((z_non_draws==results_F1).astype(int))
#


# import matplotlib.pyplot as plt
# plt.plot(valid_costs)
# plt.show()
