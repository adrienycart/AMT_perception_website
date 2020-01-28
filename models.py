import codecs
import os
import random
import cPickle as pickle

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
import pandas as pd

from config import MAX_ANSWERS


BATCH_SIZE = 100


def get_feature_labels(features_to_use):

    def get_feat_names(feat):
        if feat == 'framewise':
            return ["framewise_P","framewise_R","framewise_F"]
        elif 'notewise_On_' in feat:
            return ["notewise_On_P",
            "notewise_On_R",
            "notewise_On_F"]
        elif 'notewise_OnOff_' in feat:
            return [    "notewise_OnOff_P",
                "notewise_OnOff_R",
                "notewise_OnOff_F"]
        elif feat == "high_f":
            return ["high_f_P",
            "high_f_R",
            "high_f_F",]
        elif feat == "low_f":
            return ["low_f_P",
            "low_f_R",
            "low_f_F",]
        elif feat == "high_n":
            return ["high_n_P",
            "high_n_R",
            "high_n_F",]
        elif feat == "low_n":
            return ["low_n_P",
            "low_n_R",
            "low_n_F"]
        elif feat == "loud_fn":
            return ["loud_fn"]
        elif feat == "loud_ratio_fn":
            return ["loud_ratio_fn"]
        elif feat == "out_key":
            return ["out_key_fp",
            "out_key_all",]
        elif feat == "out_key_bin":
            return ["out_key_bin_fp",
            "out_key_bin_all",]
        elif feat == "repeat":
            return ["repeat_fp",
            "repeat_all",]
        elif feat == "merge":
            return ["merge_fp",
            "merge_all",]
        elif feat == "semitone_f":
            return ["semitone_f_fp",
            "semitone_f_all"]
        elif feat == "octave_f":
            return ["octave_f_fp",
            "octave_f_all"]
        elif feat == "third_harmonic_f":
            return ["third_harmonic_f_fp",
            "third_harmonic_f_all",]
        elif feat == "semitone_n":
            return ["semitone_n_fp",
            "semitone_n_all",]
        elif feat == "octave_n":
            return ["octave_n_fp",
            "octave_n_all",]
        elif feat == "third_harmonic_n":
            return ["third_harmonic_n_fp",
            "third_harmonic_n_all",]

    return sum([get_feat_names(feat) for feat in features_to_use],[])

def get_y_z_alphas(ratings):

    y = (ratings==int(round(MAX_ANSWERS/2))).astype(int)
    z = (ratings>int(round(MAX_ANSWERS/2))).astype(int)
    # alpha is between 0 and 0.5
    alphas = np.abs(ratings-int(round(MAX_ANSWERS/2)))/int(round(MAX_ANSWERS))

    return y,z,alphas

def linear_regression_model(features,weights,pca_matrix=None):
    #features_o and features_t are of shape: [batch_size, n_features]

    if pca_matrix is None:
        output = tf.sigmoid(tf.matmul(features, weights))
    else:
        output = tf.sigmoid(tf.matmul(tf.matmul(features,tf.cast(pca_matrix,tf.float32),transpose_b=True), weights))

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
           (1-y)*(z*tf.square(batch1) + (1-z)*tf.square(batch2))
    return tf.reduce_mean(loss)

def contrastive_loss_absolute(batch1,batch2,y,z,alphas):
    # y[i] = 1 iff batch1[i] and batch2[i] were rated equally
    # z[i] = 1 iff batch1[i] was better rated than batch2[i]

    loss = y * tf.square(batch1 - batch2) + \
           (1-y)*tf.square(tf.maximum(alphas-z*(batch2-batch1)-(1-z)*(batch1-batch2),0))
    return tf.reduce_mean(loss)

def import_features(results,features_to_use):

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

def split_data(ranges,*args):
    output = []
    for arg in args:
        out = []
        for start,end in ranges:
            out += [arg[start:end]]
        output += [np.concatenate(out,axis=0)]
    return output

#### Prepare data:
feature_dir = 'precomputed_features'

filecp = codecs.open('db_csv/answers_data.csv', encoding = 'utf-8')
answers = np.genfromtxt(filecp,dtype=object,delimiter=";")
answers = answers[1:,:]

filecp = codecs.open('db_csv/user_data.csv', encoding = 'utf-8')
users = np.genfromtxt(filecp,dtype=object,delimiter=";")
users = users[1:,:]


# #################################################
# ###### Bradley-Terry comparison : DOESN'T WORK
# #################################################
#
# import choix
# # systems = ['cheng','google',"kelz","lisu"]
# syst_dict = {'cheng':0,'google':1,"kelz":2,"lisu":3}
# data = []
# for row in answers:
#     choice = int(row[5])
#     systs=row[2:4]
#     data += [(syst_dict[systs[choice]],syst_dict[systs[1-choice]])]
#
# n_items = 4
# params = choix.ilsr_pairwise(n_items, data)
# print params
# print("ranking (worst to best):", np.argsort(params))

#################################################@
####### MIXED EFFECTS MODEL
#################################################@

#
# AQ = pd.read_csv('db_csv/answers_data.csv',delimiter=';')
#
# results_dict = {}
# feature_dir = 'precomputed_features'
#
# systems = ['cheng','google',"kelz","lisu"]
# pairs = []
# for i in range(len(systems)):
#     for j in range(i+1,len(systems)):
#         pairs += [[systems[i],systems[j]]]
#
# r = np.array(list(range(len(pairs))))
#
# for example in np.unique(answers[:,1]):
#     example_dir = os.path.join(feature_dir,example)
#     results_dict[example] = {}
#     for system in systems:
#         results = pickle.load(open(os.path.join(example_dir,system+'.pkl'), "rb"))
#         results_dict[example][system]=results
#
# f_syst1 = []
# f_syst2 = []
# f_diff = []
# diff_f_highest = []
# diff_repeated = []
# best_highest = []
# best_repeated = []
# for row in answers:
#     f_syst1 += [results_dict[row[1]][row[2]]["notewise_On_50"][-1]]
#     f_syst2 += [results_dict[row[1]][row[3]]["notewise_On_50"][-1]]
#     f_diff += [abs(results_dict[row[1]][row[2]]["notewise_On_50"][-1]-results_dict[row[1]][row[3]]["notewise_On_50"][-1])]
#     if results_dict[row[1]][row[2]]["notewise_On_50"][-1] > results_dict[row[1]][row[3]]["notewise_On_50"][-1]:
#         best_syst = row[2]
#         worst_syst = row[3]
#     else:
#         best_syst = row[3]
#         worst_syst = row[3]
#
#     diff_f_highest += [results_dict[row[1]][best_syst]["high_n"][-1]-results_dict[row[1]][worst_syst]["high_n"][-1]]
#     diff_repeated += [results_dict[row[1]][best_syst]["repeat"][-2]-results_dict[row[1]][worst_syst]["repeat"][-2]]
#     best_highest += [results_dict[row[1]][best_syst]["high_n"][-1]]
#     best_repeated += [results_dict[row[1]][best_syst]["repeat"][-1]]
#
# data = {'f_system1':f_syst1,
#         'f_system2':f_syst2,
#         'f_diff':f_diff,
#         'diff_f_highest':diff_f_highest,
#         'diff_repeated':diff_repeated,
#         'best_highest':best_highest,
#         'best_repeated':best_repeated,}
# data_f_mes = pd.DataFrame(data)
#
#
#
# goldmsi = []
# for row in answers:
#     user_id = row[4]
#     goldmsi += [float(users[users[:,0]==user_id,5])]
# data = {'goldmsi':goldmsi}
# data_goldmsi = pd.DataFrame(data)
#
# AQ_new = pd.concat([AQ, data_f_mes,data_goldmsi],axis=1)
# reverse_idx = AQ_new['f_system1']>AQ_new['f_system2']
#
#
# system1 = AQ_new["system1"][reverse_idx]
# f_system1 = AQ_new["f_system1"][reverse_idx]
# AQ_new.loc[reverse_idx, 'system1']=AQ_new["system2"][reverse_idx]
# AQ_new.loc[reverse_idx, 'system2']=system1
# AQ_new.loc[reverse_idx, 'f_system1']=AQ_new["f_system2"][reverse_idx]
# AQ_new.loc[reverse_idx, 'f_system2']=f_system1
# AQ_new.loc[reverse_idx, 'answer']=1- AQ_new["answer"][reverse_idx]

###       0            1         2          3        4         5        6            7           8        9           10             11
### ['question_id' 'example' 'system1' 'system2' 'user_id' 'answer' 'recognised' 'difficulty' 'time'  'f_system1' , 'f_system2', 'goldmsi']

# print AQ_new[['question_id','system1','system2','answer']]





# print AQ_new[['question_id','system1','system2','answer']]

### Only confident answers
# AQ_new = AQ_new[AQ_new['difficulty']<3]
# print AQ_new


### Only kelz vs lisu
# AQ_new = AQ_new[np.logical_and(np.isin(AQ_new['system1'],['kelz','lisu']),np.isin(AQ_new['system2'],['kelz','lisu']))]

### Only musicians
# goldmsi_med = np.median(AQ_new['goldmsi'])
# AQ_new = AQ_new[AQ_new['goldmsi']>goldmsi_med]

## Co-dependent variables: *
## Random variables: +(var/user_id) --> /user_id
## Also check multiple regression (simple)
## Also check multiple logistic regression

## Mixed Effects Model
# mixed = smf.mixedlm("answer ~ f_diff+f_system2+goldmsi+recognised+difficulty", AQ_new,groups='question_id')
# mixed_fit = mixed.fit()
# print(mixed_fit.summary())

### logistic regression
# feature_columns = ["recognised","difficulty","f_system1","f_system2","goldmsi"]
# X = AQ_new.loc[:, feature_columns].values
#
# y=AQ_new.answer
# clf = LogisticRegression().fit(X, y)
# print clf.coef_
# for c,feat in zip(clf.coef_[0],feature_columns):
#     print feat, 'coef', c


###############################################################
#### INTER-RATER AGREEMENT
###############################################################

### Only confident answers
# AQ_new = AQ_new[AQ_new['difficulty']<3]
# print AQ_new

### Only keep questions with 4 ratings
# q_ids,counts = np.unique(AQ_new['question_id'],return_counts=True)
# q_ids_to_keep = q_ids[counts==4]
# answers_to_keep = np.isin(AQ_new['question_id'],q_ids_to_keep)
# AQ_new = AQ_new[answers_to_keep]
#
# f_syst1s = []
# f_syst2s = []
# F_mes_diffs = []
# inter_rater_agreements = []
# agreements_with_f = []
# avg_gold_msi = []
# std_gold_msi = []
# avg_difficulty = []
# q_ids = []
#
# for q_id in np.unique(AQ_new['question_id']):
#     q_ids += [q_id]
#     data_q = AQ_new[AQ_new['question_id']==q_id]
#     f_syst1 = np.mean(data_q['f_system1'])
#     f_syst2 = np.mean(data_q['f_system2'])
#     f_syst1s += [f_syst1]
#     f_syst2s += [f_syst2]
#     F_mes_diffs += [abs(f_syst1-f_syst2)]
#     vote = np.sum(data_q["answer"])
#     agreement = abs(vote-2)
#     inter_rater_agreements += [agreement]
#
#     f_choice = 1-int(f_syst1>f_syst2)
#     agreements_with_f += [vote] if f_choice else [4-vote]
#
#     avg_gold_msi += [np.mean(data_q['goldmsi'])]
#     std_gold_msi += [np.std(data_q['goldmsi'])]
#
#     avg_difficulty += [np.mean(data_q['difficulty'])]
#
# q_ids = np.array(q_ids)
# f_syst1s = np.array(f_syst1s)
# f_syst2s = np.array(f_syst2s)
# F_mes_diffs = np.array(F_mes_diffs)
# inter_rater_agreements = np.array(inter_rater_agreements)
# agreements_with_f = np.array(agreements_with_f)
# avg_gold_msi = np.array(avg_gold_msi)
# std_gold_msi = np.array(std_gold_msi)
# avg_difficulty = np.array(avg_difficulty)
#
# AQ_new = pd.DataFrame({ 'question_id':q_ids,
#                         'f_system1':f_syst1s,
#                         'f_system2':f_syst2s,
#                         'f_diff':f_syst2s-f_syst1s,
#                         'inter_rater_agreements':inter_rater_agreements,
#                         'agreements_with_f':agreements_with_f,
#                         'avg_gold_msi':avg_gold_msi,
#                         'std_gold_msi':std_gold_msi,
#                         'avg_difficulty':avg_difficulty,})
#
# mixed = smf.mixedlm("agreements_with_f ~ f_diff+f_system2+avg_gold_msi+std_gold_msi+avg_difficulty", AQ_new,groups='question_id')
# mixed_fit = mixed.fit()
# print(mixed_fit.summary())




########################################################
###########      USING FEATURES
########################################################


#### GATHER DATA
features1=[]
features2=[]
ratings=[]
notewise1 = []
notewise2 = []
goldmsi = []

save_destination = 'results_metric/all_features'
if not os.path.exists(save_destination):
    os.makedirs(save_destination)


features_to_use = [
                "framewise",
                "notewise_On_50",
                "notewise_OnOff_50_0.2",
                "high_f",
                "low_f",
                "high_n",
                "low_n",

                "loud_fn",
                "loud_ratio_fn",

                "out_key",
                "out_key_bin",

                "repeat",
                "merge",

                "semitone_f",
                "octave_f",
                "third_harmonic_f",
                "semitone_n",
                "octave_n",
                "third_harmonic_n",
                ]

labels = get_feature_labels(features_to_use)
N_FEATURES = len(labels)

#### Graph definition

features1_ph = tf.placeholder(tf.float32,[None,N_FEATURES])
features2_ph = tf.placeholder(tf.float32,[None,N_FEATURES])
y_ph = tf.placeholder(tf.float32,[None,1])
z_ph = tf.placeholder(tf.float32,[None,1])
alpha_ph = tf.placeholder(tf.float32,[None,1])

# weights = tf.Variable(tf.random_normal([N_FEATURES,1], stddev=0.35),name='weights')
weights = tf.Variable(tf.zeros([N_FEATURES,1]),name='weights')

model_output1 = linear_regression_model(features1_ph,weights,None)
model_output2 = linear_regression_model(features2_ph,weights,None)



loss = contrastive_loss_absolute(model_output1,model_output2,y_ph,z_ph,alpha_ph)
# loss = contrastive_loss(model_output1,model_output2,y_ph,alpha_ph)
optimize = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)


### AGGREGATE ALL ANSWERS
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

#### USE EACH INDIVIDUAL ANSWER AS TRAINING SAMPLE


for row in answers:

    example = row[1]
    system1 = row[2]
    system2 = row[3]

    # print example, system1, system2

    example_dir = os.path.join(feature_dir,example)
    # print len(import_features(example_dir,system1)),import_features(example_dir,system1)
    results1 = pickle.load(open(os.path.join(example_dir,system1+'.pkl'), "rb"))
    results2 = pickle.load(open(os.path.join(example_dir,system2+'.pkl'), "rb"))

    notewise1 += [results1["notewise_On_50"][-1]]
    notewise2 += [results2["notewise_On_50"][-1]]
    goldmsi += [float(users[users[:,0]==row[4],5])]

    features1 += [import_features(results1,features_to_use)]
    features2 += [import_features(results2,features_to_use)]
    ratings += [row[5]]

features1 = np.array(features1,dtype=float)
features2 = np.array(features2,dtype=float)
ratings = np.array(ratings,dtype=int)
notewise1 = np.array(notewise1,dtype=float)
notewise2 = np.array(notewise2,dtype=float)
goldmsi = np.array(goldmsi,dtype=float)



y=np.zeros_like(ratings)
z=ratings
alpha = answers[:,7].astype(float)
alpha[alpha==1] = 0.5
alpha[alpha==2] = 0.4
alpha[alpha==3] = 0.3
alpha[alpha==4] = 0.2
alpha[alpha==5] = 0.1

y = y[:,None]
z = z[:,None]
alpha = alpha[:,None]


# dataset = pd.DataFrame({'features1':features1,
#                         "features2":features2,
#                         "ratings":ratings,
#                         "notewise1":notewise1,
#                         "notewise2":notewise2,
#                         "goldmsi":goldmsi,
#                         "y":y,
#                         "z":z,
#                         "alpha":alpha,})

#### SPLIT DATA

all_results = {}

for fold in range(10):

    all_examples, indices = np.unique(answers[:,1],return_index=True)
    sort_idx = np.argsort(indices)
    all_examples = all_examples[sort_idx]
    example_indices = indices[sort_idx]


    n_examples = len(all_examples)
    n_test = int(0.1*n_examples)
    n_valid = int(0.1*n_examples)
    ex_idx_test_start = fold*n_test
    ex_idx_test_end= (fold+1)*n_test
    if fold == 9:
        ex_idx_valid_start = 0*n_valid
        ex_idx_valid_end = 1*n_valid
    else:
        ex_idx_valid_start = (fold+1)*n_valid
        ex_idx_valid_end = (fold+2)*n_valid


    idx_test_start = example_indices[ex_idx_test_start]
    idx_test_end = example_indices[ex_idx_test_end]
    idx_valid_start = example_indices[ex_idx_valid_start]
    idx_valid_end = example_indices[ex_idx_valid_end]

    idx_test = np.zeros([len(answers)],dtype=bool)
    idx_valid = np.zeros([len(answers)],dtype=bool)
    idx_train = np.zeros([len(answers)],dtype=bool)

    if fold == 9:
        idx_test[idx_test_start:] = True
        idx_valid[idx_valid_start:idx_valid_end] = True
        idx_train[idx_valid_end:idx_test_start] = True
    else:
        idx_test[idx_test_start:idx_test_end] = True
        idx_valid[idx_valid_start:idx_valid_end] = True
        idx_train[:idx_test_start] = True
        idx_train[idx_valid_end:] = True

    features1_train = features1[idx_train]
    features2_train = features2[idx_train]
    y_train = y[idx_train]
    z_train = z[idx_train]
    alpha_train = alpha[idx_train]
    notewise1_train = notewise1[idx_train]
    notewise2_train = notewise2[idx_train]
    goldmsi_train = goldmsi[idx_train]

    features1_valid = features1[idx_valid]
    features2_valid = features2[idx_valid]
    y_valid = y[idx_valid]
    z_valid = z[idx_valid]
    alpha_valid = alpha[idx_valid]
    notewise1_valid = notewise1[idx_valid]
    notewise2_valid = notewise2[idx_valid]

    features1_test = features1[idx_test]
    features2_test = features2[idx_test]
    y_test = y[idx_test]
    z_test = z[idx_test]
    alpha_test = alpha[idx_test]
    notewise1_test = notewise1[idx_test]
    notewise2_test = notewise2[idx_test]


    ###### Apply PCA
    # pca = PCA()
    # pca.fit(np.concatenate([features1_train,features2_train],axis=0))
    # total_variance = np.cumsum(pca.explained_variance_ratio_)
    # keep_dims = np.argmax(total_variance>0.99)
    #
    # print "keep_dims", keep_dims
    # # keep_dims = 16
    # pca = PCA(n_components=keep_dims)
    # pca.fit(np.concatenate([features1_train,features2_train],axis=0))
    # pca_matrix = pca.components_

    #### No PCA:
    # pca_matrix= None
    # keep_dims = N_FEATURES

    ###############################
    ### Remove from training set

    ### Any unsure response:
    # to_keep = alpha_train[:,0]>=0.4
    ### Any non-musician response:
    # to_keep = goldmsi_train>=np.median(goldmsi_train)
    ### Any answer that agrees with F-measure (keep only those who disagree):
    # results_F1 = (notewise1_train < notewise2_train).astype(int)
    # to_keep = np.not_equal(z_train[:,0],results_F1)
    #
    # features1_train = features1_train[to_keep]
    # features2_train = features2_train[to_keep]
    # y_train = y_train[to_keep]
    # z_train = z_train[to_keep]
    # alpha_train = alpha_train[to_keep]
    # notewise1_train = notewise1_train[to_keep]
    # notewise2_train = notewise2_train[to_keep]

    repeat_agreement = []

    feed_dict_valid = {
        features1_ph:features1_valid,
        features2_ph:features2_valid,
        y_ph:y_valid,
        z_ph:z_valid,
        alpha_ph: alpha_valid,
        }

    N_REPEATS = 100

    results_F1 = (notewise1_test < notewise2_test).astype(int)
    agreement_F1 = np.mean((z_test==results_F1).astype(int))

    for i in range(N_REPEATS):
        print "fold",fold,"repeat",i

        valid_costs = []

        best_valid = None
        best_parameters = None


        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        for i in range(3000):

            features1_batch ,features2_batch,y_batch ,z_batch ,alpha_batch = sample(BATCH_SIZE,features1,features2,y,z,alpha)
            feed_dict_train = {
                features1_ph:features1_batch,
                features2_ph:features2_batch,
                y_ph:y_batch,
                z_ph:z_batch,
                alpha_ph: alpha_batch,
                }


            sess.run(optimize, feed_dict=feed_dict_train)
            valid_cost = sess.run(loss, feed_dict=feed_dict_valid)

            #Compute agreement, removing draws:
            metrics1,metrics2 = sess.run([model_output1,model_output2],feed_dict_valid)
            result_metrics = (metrics1<metrics2).astype(int)
            valid_costs += [valid_cost]

            # plt.clf()
            # plt.scatter(features1_valid[:,5],metrics1,color='tab:blue')
            # plt.scatter(features2_valid[:,5],metrics2,color='tab:blue')
            # plt.ylim([0,1])
            # plt.xlim([0,1])
            #
            # plt.pause(0.00000001)

            # print i, valid_cost, np.mean((z_valid==result_metrics).astype(int))

            if best_valid is None or valid_cost<best_valid:
                best_parameters = sess.run(weights)

        ###### RESULTS
        #
        # print 'Best parameters:'
        # for (label,value) in zip(labels,best_parameters):
        #     print label, value


        feed_dict_test = {
            features1_ph:features1_test,
            features2_ph:features2_test,
            weights: best_parameters
            }

        ## Remove draws:
        metrics1,metrics2 = sess.run([model_output1,model_output2],feed_dict_test)
        result_metrics = (metrics1<metrics2).astype(int)

        agreement_metric = np.mean((z_test==result_metrics).astype(int))
        repeat_agreement += [agreement_metric]

        print "average agreement new metric:", np.round(agreement_metric,3)
        print "average agreement F-measure:", np.round(agreement_F1,3)
        print repeat_agreement

    results_dict = {'repeat_agreement':repeat_results,
                    'agreement_F1': agreement_F1}
    save_path = os.path.join(save_destination,'fold'+str(fold)+'.pkl')
    pickle.dump(results_dict, open(save_path, 'wb'))

    all_results['fold'+str(fold)]=results_dict

save_path = os.path.join(save_destination,'all_folds.pkl')
pickle.dump(all_results, open(save_path, 'wb'))


# plt.plot(valid_costs)
# plt.show()
#
# plt.scatter(notewise1_test,metrics1,color='tab:blue')
# plt.scatter(notewise2_test,metrics2,color='tab:blue')
# plt.ylim([0,1])
# plt.xlim([0,1])
#
# plt.show()
