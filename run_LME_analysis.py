import codecs
import os
import random
import cPickle as pickle

import numpy as np
import matplotlib.pyplot as plt

import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
import pandas as pd

from config import MAX_ANSWERS




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
diff_f_highest = []
diff_repeated = []
best_highest = []
best_repeated = []
for row in answers:
    f_syst1 += [results_dict[row[1]][row[2]]["notewise_On_50"][-1]]
    f_syst2 += [results_dict[row[1]][row[3]]["notewise_On_50"][-1]]
    f_diff += [abs(results_dict[row[1]][row[2]]["notewise_On_50"][-1]-results_dict[row[1]][row[3]]["notewise_On_50"][-1])]
    if results_dict[row[1]][row[2]]["notewise_On_50"][-1] > results_dict[row[1]][row[3]]["notewise_On_50"][-1]:
        best_syst = row[2]
        worst_syst = row[3]
    else:
        best_syst = row[3]
        worst_syst = row[3]

    diff_f_highest += [results_dict[row[1]][best_syst]["high_n"][-1]-results_dict[row[1]][worst_syst]["high_n"][-1]]
    diff_repeated += [results_dict[row[1]][best_syst]["repeat"][-2]-results_dict[row[1]][worst_syst]["repeat"][-2]]
    best_highest += [results_dict[row[1]][best_syst]["high_n"][-1]]
    best_repeated += [results_dict[row[1]][best_syst]["repeat"][-1]]

data = {'f_system1':f_syst1,
        'f_system2':f_syst2,
        'f_diff':f_diff,
        'diff_f_highest':diff_f_highest,
        'diff_repeated':diff_repeated,
        'best_highest':best_highest,
        'best_repeated':best_repeated,}
data_f_mes = pd.DataFrame(data)



goldmsi = []
for row in answers:
    user_id = row[4]
    goldmsi += [float(users[users[:,0]==user_id,5])]
data = {'goldmsi':goldmsi}
data_goldmsi = pd.DataFrame(data)

AQ_new = pd.concat([AQ, data_f_mes,data_goldmsi],axis=1)
reverse_idx = AQ_new['f_system1']>AQ_new['f_system2']


system1 = AQ_new["system1"][reverse_idx]
f_system1 = AQ_new["f_system1"][reverse_idx]
AQ_new.loc[reverse_idx, 'system1']=AQ_new["system2"][reverse_idx]
AQ_new.loc[reverse_idx, 'system2']=system1
AQ_new.loc[reverse_idx, 'f_system1']=AQ_new["f_system2"][reverse_idx]
AQ_new.loc[reverse_idx, 'f_system2']=f_system1
AQ_new.loc[reverse_idx, 'answer']=1- AQ_new["answer"][reverse_idx]

###       0            1         2          3        4         5        6            7           8        9           10             11
### ['question_id' 'example' 'system1' 'system2' 'user_id' 'answer' 'recognised' 'difficulty' 'time'  'f_system1' , 'f_system2', 'goldmsi']

# print AQ_new[['question_id','system1','system2','answer']]





# print AQ_new[['question_id','system1','system2','answer']]

### Only confident answers
AQ_new = AQ_new[AQ_new['difficulty']<3]
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
mixed = smf.mixedlm("answer ~ f_diff+f_system2+goldmsi+recognised+difficulty", AQ_new,groups='question_id')
mixed_fit = mixed.fit()
print(mixed_fit.summary())

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
