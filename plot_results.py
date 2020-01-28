import os
try:
   import cPickle as pickle
except:
   import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
# import seaborn as sns
import codecs
import sklearn
from statsmodels.stats.inter_rater import fleiss_kappa
import random
import tqdm

import warnings
warnings.filterwarnings("error")

plt.rcParams.update({'font.size': 13.5,'font.family' : 'serif'})
rc('text', usetex=True)

def bootstrap(function,data,n_repeat=100,**kwargs):
    all_results = []

    for i in tqdm.trange(n_repeat):
        new_data = sklearn.utils.resample(data,replace=True)

        result = function(new_data,**kwargs)

        all_results += [result]

    all_results = np.array(all_results)
    mean = np.mean(all_results,axis=0)
    std = np.std(all_results,axis=0)

    return mean, std

barWidth = 0.75


# ############################################
# ##### USERS
# ############################################


### Gold-MSI scores in general population: mean
### Perceptual abilities: 50.20, 9 questions --> 5.578
### Musical Training: 26.52, 7 questions --> 3.789
### Overall: 4.795

### Gold-MSI scores in general population: median
### Perceptual abilities: 50, 9 questions --> 5.556
### Musical Training: 27, 7 questions --> 3.857
### Overall: 4.813





filecp = codecs.open('db_csv/user_data.csv', encoding = 'utf-8')
users = np.genfromtxt(filecp,dtype=object,delimiter=";")
users = users[1:,:]
filecp.close()

###
change_encoding = lambda x:x.decode('utf-8')
change_encoding = np.vectorize(change_encoding)
users = change_encoding(users)
###


n_users = users.shape[0]
# #### 0         1      2     3    4          5          ....   -1
# #### user;n_answers;gender;age;disability;gold_msi_avg ....   classical
#
# male_users = users[users[:,2]=="male",:]
# female_users = users[users[:,2]=="female",:]
# other_users = users[users[:,2]=="other",:]
#
#
# ### Genre distrbution
# n_male = len(male_users)
# n_female = len(female_users)
# n_other = len(other_users)
#
# print "total", n_users , "n_male",n_male,"n_female",n_female,"n_other",n_other
#
# values = [n_male,n_female,n_other]
# plt.pie(values,labels=['Male',"Female","Non-binary"],autopct=lambda p : '{:.2f}%  ({:,.0f})'.format(p,p * sum(values)/100),colors=['tab:blue','tab:red','tab:green'])
# plt.axis('equal')
# plt.title('Gender distribution')
# plt.show()
# #
# # ### Age distribution
# age_male = male_users[:,3]
# age_female = female_users[:,3]
# age_other = other_users[:,3]
# sns.kdeplot(age_male,shade=True,color='tab:blue',label='Male')
# sns.kdeplot(age_female,shade=True,color='tab:red',label="Female")
# # sns.kdeplot(age_other,shade=True,color='tab:green')
# plt.legend()
# plt.title('Age distribution')
# plt.show()
#
# average_age = np.mean(users[:,3].astype(int))
# median_age = np.median(users[:,3].astype(int))
# std_age = np.std(users[:,3].astype(int))
# print "average_age", average_age, "median_age", median_age, "std_age", std_age
#
# #
# #
# # ### GoldMSI distribution
# goldmsi_male = male_users[:,5]
# goldmsi_female = female_users[:,5]
# sns.kdeplot(goldmsi_male,shade=True,color='tab:blue',label='Male')
# sns.kdeplot(goldmsi_female,shade=True,color='tab:red',label="Female")
# # sns.kdeplot(age_other,shade=True,color='tab:green')
# plt.legend()
# plt.title('GoldMSI score distribution')
# plt.show()
#
# average_goldmsi = np.mean(users[:,5].astype(float))
# median_goldmsi = np.median(users[:,5].astype(float))
# std_goldmsi = np.std(users[:,5].astype(float))
# print "average_goldmsi", average_goldmsi, "median_goldmsi", median_goldmsi, "std_goldmsi", std_goldmsi
#
# # ### Number of answers distrbution
# answers_male = male_users[:,1].astype(int)
# answers_female = female_users[:,1].astype(int)
# sns.distplot(answers_male,kde=False,color='tab:blue',label='Male')
# sns.distplot(answers_female,kde=False,color='tab:red',label="Female")
# # sns.kdeplot(age_other,shade=True,color='tab:green')
# plt.legend()
# plt.title('Number of answers distribution')
# plt.show()
#
# average_answers = np.mean(users[:,1].astype(int))
# median_answers = np.median(users[:,1].astype(int))
# std_answers = np.std(users[:,1].astype(int))
# print "average_answers", average_answers, "median_answers", median_answers, "std_answers", std_answers



### Correlation classical question and average
# def get_avg_goldMSI(answers):
#     positive = np.array([1,1,0,1,0,1,1,0,1,0,0,1,1,1,1,1])
#     avg = np.mean(answers*positive + (7-answers)*(1-positive))
#     return avg
#
# all_averages = []
# all_last_question = []
#
# for row in users:
#     all_averages += [get_avg_goldMSI(row[6:-1].astype(int))]
#     all_last_question += [int(row[7])]
#
# from scipy.stats import linregress
# slope_n, intercept_n, r_value_n, p_value_n, std_err_n = linregress(all_averages,all_last_question)
#
# print("Slope = ",slope_n ,"R value = ", r_value_n,"P value = ", p_value_n,"Standard error = ", std_err_n)
# print('R2 = ', r_value_n**2)

# plt.scatter(all_averages,all_last_question)
# plt.show()
#
# data = [F_mes_diffs[agreements==0],F_mes_diffs[agreements==1],F_mes_diffs[agreements==2]]
# plt.violinplot(data, positions=[0,1,2], vert=True, widths=0.3,
#                        showextrema=True, showmedians=True)






############################################
##### PAIRWISE RESPONSES
############################################












filecp = codecs.open('db_csv/answers_data.csv', encoding = 'utf-8')
answers = np.genfromtxt(filecp,dtype=object,delimiter=";")
answers = answers[1:,:]
filecp.close()

###
answers = change_encoding(answers)
###


####       0            1         2          3        4         5        6            7           8        9           10
#### ['question_id' 'example' 'system1' 'system2' 'user_id' 'answer' 'recognised' 'difficulty' 'time'  'F_syst1' , 'F_syst2']


results_dict = {}
feature_dir = 'precomputed_features'

systems = ['cheng','google',"kelz","lisu"]
system_names = {'cheng': 'NMF', 'google':'SoA', "kelz":'CNN',"lisu":'HF'}
system_colors = {'cheng': 'tab:green', 'google':'tab:red', "kelz":'tab:blue',"lisu":'tab:orange'}
pairs = []
for i in range(len(systems)):
    for j in range(i+1,len(systems)):
        pairs += [[systems[i],systems[j]]]

r = np.array(list(range(len(pairs))))

for example in np.unique(answers[:,1]):
    example_dir = os.path.join(feature_dir,example)
    results_dict[example] = {}
    for system in systems:
        file_handle = open(os.path.join(example_dir,system+'.pkl'), "rb")
        try:
            # Python 2
            results = pickle.load(file_handle)
        except:
            # Python 3
            results = pickle.load(file_handle, encoding='latin1')
        file_handle.close()
        results_dict[example][system]=results

pairs_f1 = []

for pair in pairs:
    f1_agreement_each = 0
    # For each answer
    data = answers[np.logical_and(answers[:,2]==pair[0], answers[:,3]==pair[1])]
    f1_syst1 = []
    f1_syst2 = []
    for row in data:
        choice = int(row[5])
        f_syst1 = results_dict[row[1]][row[2]]['notewise_On_50'][-1]
        f_syst2 = results_dict[row[1]][row[3]]['notewise_On_50'][-1]
        f1_syst1 += [f_syst1]
        f1_syst2 += [f_syst2]

    pairs_f1 += [["{0:.1f}".format(100*np.mean(f1_syst1)),"{0:.1f}".format(100*np.mean(f1_syst2))]]

labels = [system_names[pair[0]]+" - "+system_names[pair[1]] for pair in pairs]
labels_f1 = [" - ".join(pair) for pair in pairs_f1]


### Pairwise comparisons


def pairwise_comparison(data,with_majority=True,with_difficulty=True):
    avg_choice = np.mean(data[:,5].astype(int))

    avg_difficulty = np.mean(data[:,7].astype(int))
    avg_time = np.mean(data[:,8].astype(float))

    output = [avg_choice,avg_difficulty,avg_time]
    if with_majority:
        # Majority: 1st choice elected, null, 2nd choice elected
        majority = [0,0,0]
        for q_id in np.unique(data[:,0]):
            vote = np.sum(data[data[:,0]==q_id,5].astype(int))
            if vote < 2:
                majority[0] += 1
            elif vote == 2:
                majority[1] += 1
            elif vote > 2:
                majority[2] += 1
        output += [majority]
    if with_difficulty:
        difficulties = [np.sum((data[:,7]==str(i)).astype(int)) for i in range(1,6)]
        output += [difficulties]

    return output

# dict_stats = {}
#
# for pair in pairs:
#     data = answers[np.logical_and(answers[:,2]==pair[0], answers[:,3]==pair[1])]
#
#     result = pairwise_comparison(data)
#     # result_bootstrap = bootstrap(pairwise_comparison,data,with_majority=False,with_difficulty=False)
#
#     dict_stats[str(pair)] = result
#
# for key in dict_stats.keys():
#     print key, dict_stats[key]
#
#
#
#
# choices = []
# difficulty = []
# majority = []
# for pair in pairs:
#     stats = dict_stats[str(pair)]
#     choices += [[stats[0],1-stats[0]]]
#     difficulty += [stats[4]]
#     majority += [stats[3]]
#
# choices = np.array(choices)
# difficulty = np.array(difficulty)
# majority = np.array(majority)






#### Plot choices
#
# plt.barh(r, 1-choices[:,0], color='tab:blue', edgecolor='black', height=barWidth)
# plt.barh(r, 1-choices[:,1], left=1-choices[:,0], color='tab:red', edgecolor='black', height=barWidth)
# for i in range(len(pairs)):
#     plt.text(-0.05,i+0.15,pairs[i][0],ha='right', va='center')
#     plt.text(-0.05,i-0.15,'F1: '+pairs_f1[i][0],ha='right', va='center',fontsize=9)
#     plt.text(1.05,i+0.15,pairs[i][1],ha='left', va='center')
#     plt.text(1.05,i-0.15,'F1: '+pairs_f1[i][1],ha='left', va='center',fontsize=9)
# frame1 = plt.gca()
# frame1.axes.yaxis.set_visible(False)
# plt.grid(color='grey', linestyle='-', linewidth=1,axis='x')
# frame1.set_axisbelow(True)
# plt.box(False)
#
# plt.title('Proportion of choices among all answers')
#
# plt.show()

### Plot choices with error bars
# dict_stats_bootstrap = {}
# for pair in pairs:
#     data = answers[np.logical_and(answers[:,2]==pair[0], answers[:,3]==pair[1])]
#     result_bootstrap = bootstrap(pairwise_comparison,data,with_majority=False,with_difficulty=False)
#     dict_stats_bootstrap[str(pair)] = result_bootstrap
# choices = []
# stds = []
# for pair in pairs:
#     stats = dict_stats_bootstrap[str(pair)]
#     print stats
#     choices += [[stats[0][0],1-stats[0][0]]]
#     stds += [[stats[1][0]]]
#
#
# choices = np.array(choices)
# stds = np.array(stds)
#
# for i in range(len(pairs)):
#     print pairs[i], 1-choices[i,0]
#     plt.barh(i, 1-choices[i,0], xerr=stds[i],capsize=2, color='tab:blue', edgecolor='black', height=barWidth)
#     plt.barh(i, 1-choices[i,1], left=1-choices[i,0], color='tab:red', edgecolor='black', height=barWidth)
#
# ### With F-measures
# # for i in range(len(pairs)):
# #     plt.text(-0.05,i+0.15,pairs[i][0],ha='right', va='center')
# #     plt.text(-0.05,i-0.15,'F1: '+pairs_f1[i][0],ha='right', va='center',fontsize=9)
# #     plt.text(1.05,i+0.15,pairs[i][1],ha='left', va='center')
# #     plt.text(1.05,i-0.15,'F1: '+pairs_f1[i][1],ha='left', va='center',fontsize=9)
# ### Without F-measures:
# for i in range(len(pairs)):
#     plt.text(-0.03,i,system_names[pairs[i][0]],ha='right', va='center')
#     plt.text(1.03,i,system_names[pairs[i][1]],ha='left', va='center')
#
# frame1 = plt.gca()
# frame1.axes.yaxis.set_visible(False)
# plt.grid(color='grey', linestyle='-', linewidth=1,axis='x')
# frame1.set_axisbelow(True)
# plt.box(False)
#
# # plt.title('Proportion of choices among all answers')
# plt.tight_layout(rect=[0.1,0,0.9,1])
# plt.show()

#
#
# #### Plot difficulties
#
# normalized_difficulty= difficulty/np.sum(difficulty,axis=1).astype(float)[:,None]
# start = np.zeros_like(normalized_difficulty[:,0])
# for i in range(5):
#     end = start + normalized_difficulty[:,i]
#     plt.barh(r, normalized_difficulty[:,i], left=np.sum(normalized_difficulty[:,:i],axis=1), color=np.array([1.0,1,1])-(i+1)/5.0*np.array([0,1,1]), edgecolor='black', height=barWidth,label=i+1)
#
#
# for i in range(len(pairs)):
#     plt.text(-0.05,i,labels[i],ha='right', va='center',fontsize=17)
#     # plt.text(-0.05,i-0.15,'F1s: '+labels_f1[i],ha='right', va='center',fontsize=8)
#
#
# frame1 = plt.gca()
# plt.grid(color='grey', linestyle='-', linewidth=1,axis='x')
# frame1.axes.yaxis.set_visible(False)
# frame1.set_axisbelow(True)
# frame1.tick_params(axis='both', which='major', labelsize=15)
# plt.box(False)
# plt.tight_layout(rect=[0.15, 0, 1, 1])
# plt.legend(bbox_to_anchor=(0.96,0.5))

# plt.title('Proportion of difficulty ratings', x=0.4)
# plt.show()
#
# ### Plot majority
#
# normalized_majority= majority/np.sum(majority,axis=1).astype(float)[:,None]
# start = np.zeros_like(normalized_majority[:,0])
# colors = ['tab:blue','grey','tab:red']
# color_labels = ['1st best','Draw','2nd best']
# for i in range(3):
#     end = start + normalized_majority[:,i]
#     plt.barh(r, normalized_majority[:,i], left=np.sum(normalized_majority[:,:i],axis=1), color=colors[i], edgecolor='black', height=barWidth,label=color_labels[i])
#
# for i in range(len(pairs)):
#     plt.text(-0.05,i+0.15,pairs[i][0],ha='right', va='center')
#     plt.text(-0.05,i-0.15,'F1: '+pairs_f1[i][0],ha='right', va='center',fontsize=9)
#     plt.text(1.05,i+0.15,pairs[i][1],ha='left', va='center')
#     plt.text(1.05,i-0.15,'F1: '+pairs_f1[i][1],ha='left', va='center',fontsize=9)
#
# frame1 = plt.gca()
# plt.grid(color='grey', linestyle='-', linewidth=1,axis='x')
# frame1.axes.yaxis.set_visible(False)
# frame1.set_axisbelow(True)
# plt.box(False)
#
# plt.legend(loc=[0,1], ncol=3)
#
# plt.title('Proportion of majority decisions for each question',y=1.1)
# plt.tight_layout(rect=[0.1, 0, 0.9, 1])
#
# plt.show()







############################################
##### COMPARISON F-MEASURE - ANSWERS
############################################

#
# dict_stats = {}
# for pair in pairs:
#
#     choices = [0,0]
#     f1_choices = [0,0,0]
#     majorities = [0,0,0]
#
#     f1_agreement_each = 0
#     # For each answer
#     data = answers[np.logical_and(answers[:,2]==pair[0], answers[:,3]==pair[1])]
#
#     for row in data:
#         choice = int(row[5])
#         f_syst1 = results_dict[row[1]][row[2]]['notewise_On_50'][-1]
#         f_syst2 = results_dict[row[1]][row[3]]['notewise_On_50'][-1]
#         f_choice = 1-int(f_syst1>f_syst2)
#
#         # print f_syst1, f_syst2, f_choice, choice
#         choices[choice] += 1
#         if f_choice == 0:
#             f1_choices[0] += 1
#         else:
#             f1_choices[2] += 1
#
#         if choice == f_choice:
#             f1_agreement_each += 1
#
#     # # Majority
#
#     f1_agreement_majority = 0
#     total_questions_not_draw = 0
#     f1_syst1 = []
#     f1_syst2 = []
#     for q_id in np.unique(data[:,0]):
#         data_q = data[data[:,0]==q_id,:]
#
#         fs_syst1 = results_dict[data_q[0,1]][data_q[0,2]]['notewise_On_50'][-1]
#         fs_syst2 = results_dict[data_q[0,1]][data_q[0,3]]['notewise_On_50'][-1]
#         f_choice = 1-int(f_syst1>f_syst2)
#
#         vote = np.sum(data_q[:,5].astype(int))
#
#         f1_syst1 += [f_syst1]
#         f1_syst2 += [f_syst2]
#
#         if vote < 2:
#             total_questions_not_draw+=1
#             majorities[0] += 1
#             if f_choice == 0:
#                 f1_agreement_majority+=1
#         elif vote == 2:
#             majorities[1] += 1
#         elif vote > 2:
#             majorities[2] += 1
#             total_questions_not_draw+=1
#             if f_choice == 1:
#                 f1_agreement_majority+=1
#
#         # print f1_syst1, np.mean(f1_syst1)
#         dict_stats[str(pair)] = [f1_agreement_each/float(data.shape[0]),f1_agreement_majority/float(total_questions_not_draw),choices,f1_choices,majorities]
#
# for key in dict_stats.keys():
#     print key, dict_stats[key]
#
# f1_agreement_each = []
# f1_agreement_majority = []
# choices = []
# majority = []
# f1_choices = []
# for pair in pairs:
#     stats = dict_stats[str(pair)]
#     f1_agreement_each += [[stats[0],1-stats[0]]]
#     f1_agreement_majority += [[stats[1],1-stats[1]]]
#     choices += [stats[2]]
#     f1_choices += [stats[3]]
#     majority += [stats[4]]
#
# f1_agreement_each = np.array(f1_agreement_each)
# f1_agreement_majority = np.array(f1_agreement_majority)
#



#### Plot choices with F-measure

# normalized_choices = choices/np.sum(choices,axis=1).astype(float)[:,None]
# normalized_f1_choices = f1_choices/np.sum(f1_choices,axis=1).astype(float)[:,None]
#
#
# plt.barh(r-barWidth/4, normalized_choices[:,0], color='tab:blue', edgecolor='black', height=barWidth/2)
# plt.barh(r-barWidth/4, normalized_choices[:,1], left=normalized_choices[:,0], color='tab:red', edgecolor='black', height=barWidth/2)
#
# plt.barh(r+barWidth/4, normalized_f1_choices[:,0], color='tab:blue', edgecolor='black', height=barWidth/2)
# plt.barh(r+barWidth/4, normalized_f1_choices[:,2], left=normalized_f1_choices[:,0], color='tab:red', edgecolor='black', height=barWidth/2)
#
# for i in range(len(pairs)):
#     plt.text(-0.05,i,pairs[i][0],ha='right', va='center')
#     plt.text(1.05,i,pairs[i][1],ha='left', va='center')
#
# frame1 = plt.gca()
# frame1.axes.yaxis.set_visible(False)
# plt.grid(color='grey', linestyle='-', linewidth=1,axis='x')
# frame1.set_axisbelow(True)
# plt.box(False)
#
# plt.title('Preference for all examples (top: F-measure, bottom: all answers)')
#
# plt.show()

### Plot majority with F-measure

# normalized_majority = majority/np.sum(majority,axis=1).astype(float)[:,None]
#
#
# colors = ['tab:blue','grey','tab:red']
# color_labels = ['1st best','Draw','2nd best']
# for i in range(3):
#     plt.barh(r-barWidth/4, normalized_majority[:,i], left=np.sum(normalized_majority[:,:i],axis=1), color=colors[i], edgecolor='black', height=barWidth/2,label=color_labels[i])
#     plt.barh(r+barWidth/4, normalized_f1_choices[:,i],left=np.sum(normalized_f1_choices[:,:i],axis=1), color=colors[i], edgecolor='black', height=barWidth/2)
#
#
# for i in range(len(pairs)):
#     plt.text(-0.05,i,pairs[i][0],ha='right', va='center')
#     plt.text(1.05,i,pairs[i][1],ha='left', va='center')
#
# frame1 = plt.gca()
# frame1.axes.yaxis.set_visible(False)
# plt.grid(color='grey', linestyle='-', linewidth=1,axis='x')
# frame1.set_axisbelow(True)
# plt.box(False)
#
# plt.legend(loc=[0,1], ncol=3)
#
# plt.title('Preference for all examples (top: F-measure, bottom: majority voting)',y=1.1)
# plt.tight_layout(rect=[0.1, 0, 0.9, 1])
#
# plt.show()



############################################
##### AGREEMENT F-MEASURE - ANSWERS
############################################

onset_tolerances = [25,50,75,100,125,150]
offset_tolerances = [0.1,0.2,0.3,0.4,0.5]

all_measures = ['framewise']+['notewise_On_'+str(on_tol) for on_tol in onset_tolerances]+['notewise_OnOff_'+str(on_tol)+'_'+str(off_tol) for on_tol in onset_tolerances for off_tol in offset_tolerances]


def agreement_f_measure_answers(data,measures=all_measures,with_majority=True,average=False):
    f1_agreement_each_all = []
    f1_agreement_majority_all = []

    for measure in measures:
        f1_agreement_each = 0
        for row in data:
            choice = int(row[5])
            f_syst1 = results_dict[row[1]][row[2]][measure][-1]
            f_syst2 = results_dict[row[1]][row[3]][measure][-1]

            f_choice = 1-int(f_syst1>f_syst2)

            if choice == f_choice:
                f1_agreement_each += 1
        f1_agreement_each_all += [[f1_agreement_each,float(data.shape[0])]]

    if average:
        output = [np.array(f1_agreement_each_all)[:,0]/np.array(f1_agreement_each_all)[:,1]]
    else:
        output = [np.array(f1_agreement_each_all)]

    if with_majority:
        # # Majority
        for measure in all_measures:
            f1_agreement_majority = 0
            total_questions_not_draw = 0
            for q_id in np.unique(data[:,0]):
                data_q = data[data[:,0]==q_id,:]

                f_syst1 = results_dict[data_q[0,1]][data_q[0,2]][measure][-1]
                f_syst2 = results_dict[data_q[0,1]][data_q[0,3]][measure][-1]
                f_choice = 1-int(f_syst1>f_syst2)


                vote = np.sum(data_q[:,5].astype(int))


                if vote < 2:
                    total_questions_not_draw+=1
                    if f_choice == 0:
                        f1_agreement_majority+=1
                elif vote == 2:
                    pass
                elif vote > 2:
                    total_questions_not_draw+=1
                    if f_choice == 1:
                        f1_agreement_majority+=1

            f1_agreement_majority_all += [[f1_agreement_majority,float(total_questions_not_draw)]]
        if average:
            output += [np.array(f1_agreement_majority_all)[:,0]/np.array(f1_agreement_majority_all)[:,1]]
        else:
            output += [np.array(f1_agreement_majority_all)]
    return output

################
with_error_bars=True
#
# ### ONLY FOR VERY CONFIDENT ANSWERS:
# answers=answers[answers[:,7].astype(int)<3]
################

# dict_stats = {}
# for pair in pairs:
#
#     # For each answer
#     data = answers[np.logical_and(answers[:,2]==pair[0], answers[:,3]==pair[1])]
#     # For each answern, dropping difficulty==5
#     # data = data[data[:,7].astype(int)<4]
#
#     if not with_error_bars:
#         dict_stats[str(pair)] = agreement_f_measure_answers(data)
#     else:
#         dict_stats[str(pair)] = bootstrap(agreement_f_measure_answers,data,n_repeat=100,with_majority=False,average=True)
#
#
# f1_agreement_each=[]
# f1_agreement_majority=[]
# if with_error_bars:
#     f1_agreement_each_std = []
# for pair in pairs:
#     if not with_error_bars:
#         f1_agreement_each += [dict_stats[str(pair)][0][:,0]/dict_stats[str(pair)][0][:,1]]
#         f1_agreement_majority += [dict_stats[str(pair)][1][:,0]/dict_stats[str(pair)][1][:,1]]
#     else:
#         f1_agreement_each += [dict_stats[str(pair)][0][0]]
#         f1_agreement_each_std += [dict_stats[str(pair)][1][0]]
# f1_agreement_each = np.array(f1_agreement_each)
# f1_agreement_majority = np.array(f1_agreement_majority)
# if with_error_bars:
#     f1_agreement_each_std = np.array(f1_agreement_each_std)

# ######
# ## ON-NOTEWISE ONLY
# f1_agreement_each = f1_agreement_each[:,1:5]
# f1_agreement_majority = f1_agreement_majority[:,1:5]
# bar_labels = ['25ms','50ms','75ms','100ms']
# n_bars = f1_agreement_each.shape[1]
# colors = [np.array([1.0,1,1])-(i+1)/float(n_bars)*np.array([0,1,1])for i in range(n_bars)]

######
## ON-NOTEWISE and FRAMEWISE
# f1_agreement_each = f1_agreement_each[:,:5]
# f1_agreement_majority = f1_agreement_majority[:,:5]
# bar_labels = ['Frame','On\n25ms','On\n50ms','On\n75ms','On\n100ms']
# n_bars = f1_agreement_each.shape[1]
# colors = ['tab:green']+[np.array([1.0,1,1])-(i+1)/float(n_bars)*np.array([0,1,1])for i in range(4)]



######
## ONOFF-NOTEWISE ONLY
# f1_agreement_each = f1_agreement_each[:,5:]
# f1_agreement_majority = f1_agreement_majority[:,5:]
# bar_labels = ['OnOff\n25ms','OnOff\n50ms','OnOff\n75ms','OnOff\n100ms']
# n_bars = f1_agreement_each.shape[1]
# colors = [np.array([1.0,1,1])-(i+1)/float(n_bars)*np.array([1,1,0])for i in range(n_bars)]

# ######
# ## ONOFF-NOTEWISE and FRAMEWISE
# f1_agreement_each = f1_agreement_each[:,[0,5,6,7,8]]
# f1_agreement_majority = f1_agreement_majority[:,[0,5,6,7,8]]
# bar_labels = ['Frame','OnOff\n25ms','OnOff\n50ms','OnOff\n75ms','OnOff\n100ms']
# n_bars = f1_agreement_each.shape[1]
# colors = ['tab:green']+[np.array([1.0,1,1])-(i+1)/float(n_bars)*np.array([1,1,0])for i in range(n_bars)]

######
## ON-NOTEWISE , ONOFF-notewise and FRAMEWISE
# f1_agreement_each = f1_agreement_each[:,[0,2,13]]
# if with_error_bars:
#     f1_agreement_each_std = f1_agreement_each_std[:,[0,2,13]]
# # f1_agreement_majority = f1_agreement_majority[:,:5]
# bar_labels = [r'$F_f$',r'$F_{n,On}$',r'$F_{n,OnOff}$']
# n_bars = f1_agreement_each.shape[1]
# colors = ['tab:green','tab:red','tab:blue']
#
#
#
# single_barWidth = barWidth/n_bars
#
#
# # # ### Plot for all answers
#
# if with_error_bars:
#     for i in range(n_bars):
#         plt.barh(r-single_barWidth/2+(i-1)*single_barWidth, f1_agreement_each[:,i],
#             xerr=f1_agreement_each_std[:,i],capsize=2,
#             color=colors[i], edgecolor='black',
#             height=single_barWidth,label=bar_labels[i])
# else:
#     for i in range(n_bars):
#         plt.barh(r-single_barWidth/2+(i-1)*single_barWidth, f1_agreement_each[:,i],
#             color=colors[i], edgecolor='black',
#             height=single_barWidth,label=bar_labels[i])
#
# frame1 = plt.gca()
# plt.yticks(r,labels)
# plt.grid(color='grey', linestyle='-', linewidth=1,axis='x')
# frame1.set_axisbelow(True)
# plt.box(False)
# plt.xlim((0,1.1))
# # plt.title('Agreement between raters and F-measure,\n with various onset thresholds (all answers)')
# plt.tight_layout(rect=[0, 0, 1, 1])
# plt.legend()
# plt.show()

# ### Plot for majority voting only
#
# for i in range(n_bars):
#     plt.barh(r-single_barWidth/2+(i-1)*single_barWidth, f1_agreement_majority[:,i],
#         color=colors[i], edgecolor='black',
#         height=single_barWidth,label=bar_labels[i])
#
# frame1 = plt.gca()
# plt.yticks(r,labels)
# plt.grid(color='grey', linestyle='-', linewidth=1,axis='x')
# frame1.set_axisbelow(True)
# plt.box(False)
# plt.xlim((0,1.1))
# plt.legend()
# plt.title('Agreement between raters and F-measure,\n with various onset thresholds (majority voting, no draws)',x=0.4)
# plt.tight_layout(rect=[0, 0, 1, 1])
# plt.show()

#### ACROSS ALL SYSTEM PAIRS, WITH ERROR bars

# ### ONLY FOR VERY CONFIDENT ANSWERS:
# answers=answers[answers[:,7].astype(int)<3]

# f1_agreement_each,f1_agreement_each_std = bootstrap(agreement_f_measure_answers,answers,n_repeat=100,with_majority=False,average=True)
# f1_agreement_each = f1_agreement_each[0]
# f1_agreement_each_std = f1_agreement_each_std[0]
#
# labels_on = [r'$F_{n,On}$'+'\n'+str(on_tol)+'ms' for on_tol in onset_tolerances]
# labels_on_off = ['OnOff\n'+str(on_tol)+'ms\n'+str(off_tol*100)+'%' for on_tol in onset_tolerances for off_tol in offset_tolerances]
# labels = [r'$F_f$']+labels_on+labels_on_off
#
# colors = ['tab:green']+[np.array([1.0,1,1])-(i+1)/float(len(labels_on))*np.array([0,1,1]) for i in range(len(labels_on))]+[np.array([1.0,1,1])-(i+1)/float(len(labels_on_off))*np.array([1,1,0]) for i in range(len(labels_on_off))]


### Only On-notewise metrics:
# lab = labels[:len(labels_on)+1]
# col = colors[:len(labels_on)+1]
# values = f1_agreement_each[:len(labels_on)+1]
# stds = f1_agreement_each_std[:len(labels_on)+1]
#
# plt.bar(list(range(len(values))), values, yerr=stds,capsize=2,color=col, edgecolor='black', width=barWidth)
# frame1 = plt.gca()
# plt.xticks(list(range(len(values))),lab)
# plt.grid(color='grey', linestyle='-', linewidth=1,axis='y')
# frame1.set_axisbelow(True)
# plt.box(False)
# plt.ylim((0.7,0.84))
# # plt.title('Agreement between raters and various F-measures\n(all answers)')
# plt.ylabel(r'Agreement with ratings',fontsize=15)
# plt.tight_layout(rect=[0, 0, 1, 1])
# plt.show()

######
## ON-notewise , ONOFF-notewise and framewise
# idxs = [0,2,13]
# lab = np.array(labels,dtype=object)[idxs]
# col = ['tab:green','tab:red','tab:blue']
# values = np.array(f1_agreement_each,dtype=object)[idxs]
# stds = np.array(f1_agreement_each_std,dtype=object)[idxs]
#
#
# plt.bar(list(range(len(values))), values, yerr=stds,capsize=2,color=col, edgecolor='black', width=barWidth)
# frame1 = plt.gca()
# plt.xticks(list(range(len(values))),lab)
# plt.grid(color='grey', linestyle='-', linewidth=1,axis='y')
# frame1.set_axisbelow(True)
# plt.box(False)
# plt.ylim((0.7,0.83))
# plt.title('Agreement between raters and various F-measures\n(all answers)')
# plt.tight_layout(rect=[0, 0, 1, 1])
# plt.show()


#
# ### On-Off Matrix
#
# matrix = np.zeros([len(onset_tolerances),len(offset_tolerances)])
#
# idx = len(onset_tolerances)+1
# for i in range(len(onset_tolerances)):
#     for j in range(len(offset_tolerances)):
#         matrix[i,j] = f1_agreement_each[idx]
#         idx+=1
#
#
# plt.imshow(matrix,cmap='inferno',aspect='auto')
# plt.xticks(range(len(offset_tolerances)),offset_tolerances)
# plt.yticks(range(len(onset_tolerances)),onset_tolerances)
# plt.ylabel("Onset tolerance (in milliseconds)", fontsize=15)
# plt.xlabel("Offset tolerance (in proportion of note duration)", fontsize=15)
# # plt.title("Agreement between raters and On-Off F-measure\nfor various onset and offset tolerance thresholds")
# cbar_ticks = [np.min(matrix),(np.min(matrix)+np.max(matrix))/2,np.max(matrix)]
# cbar_ticks_labels = np.round([np.min(matrix),(np.min(matrix)+np.max(matrix))/2,np.max(matrix)],3)
# cbar = plt.colorbar()
# cbar.set_ticks(cbar_ticks)
# cbar.set_ticklabels(cbar_ticks_labels)
#
# plt.show()

### Majorities
# plt.bar(list(range(len(f1_agreement_each))), f1_agreement_majority, color=colors, edgecolor='black', width=barWidth)
# frame1 = plt.gca()
# plt.xticks(list(range(len(f1_agreement_each))),labels)
# plt.grid(color='grey', linestyle='-', linewidth=1,axis='y')
# frame1.set_axisbelow(True)
# plt.box(False)
# plt.ylim((0,1.05))
# plt.title('Agreement between raters and various F-measures\n(majority voting, no draws)')
# plt.tight_layout(rect=[0, 0, 1, 1])
# plt.show()






############################################
##### CORRELATIONS
############################################





####       0            1         2          3        4         5        6            7           8        9           10
#### ['question_id' 'example' 'system1' 'system2' 'user_id' 'answer' 'recognised' 'difficulty' 'time'  'F_syst1' , 'F_syst2']

F_measures = []
for row in answers:
    f_syst1 = results_dict[row[1]][row[2]]["notewise_On_50"][-1]
    f_syst2 = results_dict[row[1]][row[3]]["notewise_On_50"][-1]
    F_measures += [[f_syst1,f_syst2]]

F_measures = np.array(F_measures)
# print F_measures
answers = np.concatenate([answers,F_measures],axis=1)
# print answers

#### F-measure difference vs disagreement

F_mes_diffs = []
inter_rater_agreements = []
agreements_with_f = []
agree_disagree = []
raw_votes = []

# ### ONLY FOR VERY CONFIDENT ANSWERS:
answers=answers[answers[:,7].astype(int)<3]
# ### ONLY FOR FULL QUESTIONS:
q_ids,counts = np.unique(answers[:,0],return_counts=True)
q_ids_to_keep = q_ids[counts==4]
# print q_ids_to_keep.shape
answers_to_keep = np.isin(answers[:,0],q_ids_to_keep)
answers = answers[answers_to_keep]



for q_id in np.unique(answers[:,0]):
    data_q = answers[answers[:,0]==q_id,:]
    F_mes_diffs += [abs(float(data_q[0,-1])-float(data_q[0,-2]))]
    vote = np.sum(data_q[:,5].astype(int))
    agreement = abs(vote-2)
    inter_rater_agreements += [agreement]

    f_choice = 1-int(float(data_q[0,-2])>float(data_q[0,-1]))
    agreements_with_f += [vote] if f_choice else [4-vote]
    # agree_disagree += [[4-vote, vote]] if f_choice else [[vote, 4-vote]]
    # agree_disagree += [[4-vote, vote]] if random.choice([True, False]) else [[vote, 4-vote]]

    # order = random.choice([True, False])
    order = f_choice
    if order:
        raw_votes += [[np.sum(data_q[:,5].astype(int)==0),np.sum(data_q[:,5].astype(int)==1)]]
    else:
        raw_votes += [[np.sum(data_q[:,5].astype(int)==1),np.sum(data_q[:,5].astype(int)==0)]]

F_mes_diffs = np.array(F_mes_diffs)
agreements_with_f = np.array(agreements_with_f)
raw_votes = np.array(raw_votes)


# print fleiss_kappa(np.array(raw_votes))

# import krippendorff
# print(krippendorff.alpha(value_counts=raw_votes,level_of_measurement='nominal'))


for i in range(5):
    data = F_mes_diffs[agreements_with_f==i]
    print i, np.mean((agreements_with_f==i).astype(int)), np.sum((agreements_with_f==i).astype(int)), np.mean(data)

#
# F_mes_diffs = np.array(F_mes_diffs)
# agreements = np.array(agreements)
# data = [F_mes_diffs[agreements==0],F_mes_diffs[agreements==1],F_mes_diffs[agreements==2]]
# plt.violinplot(data, positions=[0,1,2], vert=True, widths=0.3,
#                        showextrema=True, showmedians=True)
# plt.xticks([0,1,2],["Draw","3 vs 1","Unanimous"])
# plt.title("Difference in F-measure vs. inter-rater agreement")
# plt.show()
#
# #### F-measure difference vs reported difficulty
#
# F_mes_diffs = np.abs(answers[:,-1]-answers[:,-2])
# difficulties = answers[:,7]
#
# data = [F_mes_diffs[difficulties==str(i)].astype(float) for i in range(1,6)]
#
# plt.figure(figsize=(7,4))
# plt.violinplot(data, positions=range(1,6), vert=True, widths=0.3,
#                        showextrema=True, showmedians=True)
# plt.xticks(range(1,6),range(1,6))
# plt.grid(color='lightgrey', linestyle='-', linewidth=1,axis='y')
# plt.box(False)
# plt.ylabel(r'$F_{n,on}$ difference',fontsize=15)
# plt.xlabel('Reported difficulty',fontsize=15)
# # plt.title("Difference in F-measure vs. reported difficulty")
# plt.tight_layout()
# plt.show()

#### F-measure chosen option vs disagreement

# F_mes_diffs = []
# agreements = []
# for q_id in np.unique(answers[:,0]):
#     data_q = answers[answers[:,0]==q_id,:]
#     F_mes_diffs += [int(data_q[0,5])*data_q[0,-1]+(1-int(data_q[0,5]))*data_q[0,-2]]
#     vote = np.sum(data_q[:,5].astype(int))
#     agreement = abs(vote-2)
#     agreements += [agreement]
#
# F_mes_diffs = np.array(F_mes_diffs)
# agreements = np.array(agreements)
# data = [F_mes_diffs[agreements==0],F_mes_diffs[agreements==1],F_mes_diffs[agreements==2]]
# plt.violinplot(data, positions=[0,1,2], vert=True, widths=0.3,
#                        showextrema=True, showmedians=True)
# plt.xticks([0,1,2],["Draw","3 vs 1","Unanimous"])
# plt.title("F-measure of chosen option vs. inter-rater agreement")
# plt.show()
#
# #### F-measure chosen option vs reported difficulty
#
# F_mes_chosen = answers[:,5].astype(int)*answers[:,-1]+(1-answers[:,5].astype(int))*answers[:,-2]
# difficulties = answers[:,7]
#
# data = [F_mes_chosen[difficulties==str(i)].astype(float) for i in range(1,6)]
#
# plt.figure(figsize=(7,4))
# plt.violinplot(data, positions=range(1,6), vert=True, widths=0.3,
#                        showextrema=True, showmedians=True)
# plt.xticks(range(1,6),range(1,6))
# plt.grid(color='lightgrey', linestyle='-', linewidth=1,axis='y')
# plt.box(False)
# plt.ylabel(r'Chosen $F_{n,on}$',fontsize=15)
# plt.xlabel('Reported difficulty',fontsize=15)
# plt.tight_layout()
# plt.show()


#### Agreement with F-measure vs difficulty

# def f_measure_agreement_by_difficulty(data):
#     f_choices = 1-(data[:,-2]>data[:,-1]).astype(int)
#     agree = (f_choices==data[:,5].astype(int)).astype(int)
#     difficulties = data[:,7]
#     # print [(agree[difficulties[5]==str(i)]).shape for i in range(1,6)]
#     return [np.mean(agree[difficulties==str(i)]) for i in range(1,6)]
#
#
# data,std = bootstrap(f_measure_agreement_by_difficulty,answers)
#
# plt.bar([1,2,3,4,5],data,yerr=std,capsize=2,width=barWidth,edgecolor='black')
# frame1 = plt.gca()
# plt.xticks([1,2,3,4,5],[1,2,3,4,5])
# plt.grid(color='grey', linestyle='-', linewidth=1,axis='y')
# frame1.set_axisbelow(True)
# plt.box(False)
# plt.ylim((0,1.05))
# plt.ylabel(r'Agreement between ratings and $F_{n,on}$',fontsize=15)
# plt.xlabel("Reported difficulty",fontsize=15)
# # plt.title('Agreement between raters and F-measure\nfor each reported difficulty')
# plt.tight_layout(rect=[0, 0, 1, 1])
# plt.show()

#### Agreement with F-measure vs best F-measure of the pair

# ### ONLY FOR VERY CONFIDENT ANSWERS:
# answers=answers[answers[:,7].astype(int)<3]
n_bins = 10

def f_measure_agreement_by_f_measure(data,to_compare):
    f_choices = 1-(data[:,-2]>data[:,-1]).astype(int)
    agree = (f_choices==data[:,5].astype(int)).astype(int)

    if to_compare == 'chosen':
        all_fs = data[:,-2:]
        fs = all_fs[range(len(f_choices)),f_choices]

    if to_compare == 'best':
        fs = np.maximum(data[:,-1],data[:,-2])

    if to_compare == 'diff':
        fs = np.abs(data[:,-1]-data[:,-2])

    # print [(agree[difficulties[5]==str(i)]).shape for i in range(1,6)]

    return [np.mean(agree[np.logical_and(fs>i/float(n_bins),fs<(i+1)/float(n_bins))]) if np.any(np.logical_and(fs>i/float(n_bins),fs<(i+1)/float(n_bins))) else 0 for i in range(0,n_bins)]


######## Agreement vs diff

# data,std = bootstrap(f_measure_agreement_by_f_measure,answers,to_compare='chosen')
# plt.bar(np.arange(n_bins)+0.5,data,yerr=std,capsize=2,edgecolor="black",width=1)
# frame1 = plt.gca()
# plt.xticks(range(n_bins+1),[i/float(n_bins) for i in range(n_bins+1)])
# plt.grid(color='grey', linestyle='-', linewidth=1,axis='y')
# frame1.set_axisbelow(True)
# plt.box(False)
# plt.ylim((0,1.05))
# # plt.title('Agreement between raters and F-measure\ndepending on the F-measure of the chosen solution ')
# plt.ylabel(r'Agreement between ratings and $F_{n,on}$')
# plt.xlabel(r'$F_{n,on}$ of chosen option')
# # plt.tight_layout(rect=[0, 0, 1, 1])
# plt.show()

######## Agreement vs diff

# data,std = bootstrap(f_measure_agreement_by_f_measure,answers,to_compare='diff')
# plt.bar(np.arange(n_bins-1)+0.5,data[:-1],yerr=std[:-1],capsize=2,edgecolor="black",width=1)
# frame1 = plt.gca()
# plt.xticks(range(n_bins),[i/float(n_bins) for i in range(n_bins)])
# plt.grid(color='grey', linestyle='-', linewidth=1,axis='y')
# frame1.set_axisbelow(True)
# plt.box(False)
# plt.ylim((0.5,1.05))
# # plt.title('Agreement between raters and F-measure\ndepending on the F-measure of the chosen solution ')
# plt.ylabel(r'Agreement between ratings and $F_{n,on}$')
# plt.xlabel(r'Absolute difference in $F_{n,on}$')
# # plt.tight_layout(rect=[0, 0, 1, 1])
# plt.show()

#### Known vs Unknown pieces

# known = answers[:,6]=='True'
# unknown = answers[:,6]=='False'
#
# agree_known = 0
# agree_unknown = 0
# difficulty_known = [0,0,0,0,0]
# difficulty_unknown = [0,0,0,0,0]
# for row in answers:
#     f_choice = 1-int(row[-2]>row[-1])
#     agree = f_choice == int(row[5])
#
#     if row[6] == 'True':
#         difficulty_known[int(row[7])-1] += 1
#         if agree:
#             agree_known += 1
#     elif row[6] == 'False':
#         difficulty_unknown[int(row[7])-1] += 1
#         if agree:
#             agree_unknown += 1
# difficulty_known = np.array(difficulty_known)/float(sum(difficulty_known))
# difficulty_unknown = np.array(difficulty_unknown)/float(sum(difficulty_unknown))


# plt.bar([0,1],[agree_known/float(np.sum(known.astype(int))),agree_unknown/float(np.sum(unknown.astype(int)))],width=barWidth)
# frame1 = plt.gca()
# plt.xticks([0,1],['Known','Unknown'])
# plt.grid(color='grey', linestyle='-', linewidth=1,axis='y')
# frame1.set_axisbelow(True)
# plt.box(False)
# plt.ylim((0,1.05))
# plt.title('Agreement between raters and F-measure\nfor known and unknown examples')
# plt.tight_layout(rect=[0, 0, 1, 1])
# plt.show()

# difficulties = np.concatenate([difficulty_known[None,:],difficulty_unknown[None,:]],axis=0)
# print difficulties.shape
#
# start = np.zeros([2])
# for i in range(5):
#     end = start + difficulties[:,i]
#     plt.barh([0,1], difficulties[:,i], left=np.sum(difficulties[:,:i],axis=1), color=np.array([1.0,1,1])-(i+1)/5.0*np.array([0,1,1]), edgecolor='black', height=barWidth)
#
# frame1 = plt.gca()
# plt.yticks([0,1],['Known','Unknown'])
# # plt.grid(color='grey', linestyle='-', linewidth=1,axis='y')
# frame1.set_axisbelow(True)
# plt.box(False)
# # plt.ylim((0,1.05))
# plt.title('Reported difficulty for known and unknown examples')
# # plt.tight_layout(rect=[0, 0, 1, 1])
# plt.show()


#### Musicians vs non-musicians

# goldmsi = users[:,5].astype(float)
# median_goldmsi = np.median(goldmsi)
#
# musicians_id = users[goldmsi>median_goldmsi,0]
# non_musicians_id = users[goldmsi<median_goldmsi,0]
#
# agree_musicians = 0
# total_musicians = 0
# agree_non_musicians = 0
# total_non_musicians = 0
# for row in answers:
#     f_choice = 1-int(row[-2]>row[-1])
#     agree = f_choice == int(row[5])
#
#     if row[4] in musicians_id:
#         total_musicians+=1
#
#         if agree :
#             agree_musicians += 1
#     elif row[4] in non_musicians_id:
#         total_non_musicians+=1
#         if agree :
#             agree_non_musicians += 1
#
#
# plt.bar([0,1],[agree_musicians/float(total_musicians),agree_non_musicians/float(total_non_musicians)],width=barWidth,edgecolor='black')
# frame1 = plt.gca()
# plt.xticks([0,1],['Musicians','Non-musicians'])
# plt.grid(color='grey', linestyle='-', linewidth=1,axis='y')
# frame1.set_axisbelow(True)
# plt.box(False)
# plt.ylim((0,1.05))
# plt.title('Agreement between raters and F-measure\nfor musicians and non-musicians')
# plt.tight_layout(rect=[0, 0, 1, 1])
# plt.show()


#### Difficuly vs GoldMSI

# goldmsi = []
# classical = []
# avg_difficulty = []
# avg_agreement = []
# known = []
# for user in users:
#     data = answers[answers[:,4]==user[0]]
#     goldmsi += [float(user[5])]
#     classical += [int(user[-1])]
#     avg_difficulty += [np.mean(data[:,7].astype(int))]
#
#     f_choice = 1-(data[:,-2]>data[:,-1]).astype(int)
#     agree = f_choice == data[:,5].astype(int)
#     avg_agreement += [np.mean(agree.astype(int))]
#
#     known += [np.mean((data[:,6]=='True').astype(int))]


# plt.scatter(goldmsi,avg_difficulty)
# frame1 = plt.gca()
# plt.xticks(range(2,8),range(2,8))
# plt.yticks(range(6),range(6))
# plt.xlabel('GoldMSI Score')
# plt.ylabel('Average reported difficulty')
# plt.grid(color='grey', linestyle='-', linewidth=1,axis='y')
# frame1.set_axisbelow(True)
# plt.box(False)
# plt.title('Average reported difficulty vs. GoldMSI Score')
# plt.tight_layout(rect=[0, 0, 1, 1])
# plt.show()

# plt.scatter(goldmsi,known)
# frame1 = plt.gca()
# plt.xticks(range(2,8),range(2,8))
# # plt.yticks(range(6),range(6))
# plt.xlabel('GoldMSI Score')
# plt.ylabel('Proportion of recognised pieces')
# plt.grid(color='grey', linestyle='-', linewidth=1,axis='y')
# frame1.set_axisbelow(True)
# plt.box(False)
# plt.title('Proportion of recognised pieces vs. GoldMSI Score')
# plt.tight_layout(rect=[0, 0, 1, 1])
# plt.show()

# plt.scatter(classical,known)
# frame1 = plt.gca()
# plt.xticks(range(2,8),range(2,8))
# # plt.yticks(range(6),range(6))
# plt.xlabel('Listens to classical music')
# plt.ylabel('Proportion of recognised pieces')
# plt.grid(color='grey', linestyle='-', linewidth=1,axis='y')
# frame1.set_axisbelow(True)
# plt.box(False)
# plt.title('Proportion of recognised pieces vs. GoldMSI Score')
# plt.tight_layout(rect=[0, 0, 1, 1])
# plt.show()






### COMPARISON CHENG-ENSTDKCL vs CHENG-ENSTCKAM

# cl_idx = np.core.defchararray.find(answers[:,1].astype('U256'),'ENSTDkCl')!=-1
# am_idx = np.core.defchararray.find(answers[:,1].astype('U256'),'ENSTDkAm')!=-1
#
# answers_cl = answers[cl_idx]
# answers_am = answers[am_idx]
#
# pairs_f1_cl = []
# for pair in pairs:
#     f1_agreement_each = 0
#     # For each answer
#     data = answers_cl[np.logical_and(answers_cl[:,2]==pair[0], answers_cl[:,3]==pair[1])]
#     f1_syst1 = []
#     f1_syst2 = []
#     for row in data:
#         choice = int(row[5])
#         f_syst1 = results_dict[row[1]][row[2]]['notewise_On_50'][-1]
#         f_syst2 = results_dict[row[1]][row[3]]['notewise_On_50'][-1]
#         f1_syst1 += [f_syst1]
#         f1_syst2 += [f_syst2]
#     pairs_f1_cl += [["{0:.1f}".format(100*np.mean(f1_syst1)),"{0:.1f}".format(100*np.mean(f1_syst2))]]
#
# pairs_f1_am = []
# for pair in pairs:
#     f1_agreement_each = 0
#     # For each answer
#     data = answers_am[np.logical_and(answers_am[:,2]==pair[0], answers_am[:,3]==pair[1])]
#     f1_syst1 = []
#     f1_syst2 = []
#     for row in data:
#         choice = int(row[5])
#         f_syst1 = results_dict[row[1]][row[2]]['notewise_On_50'][-1]
#         f_syst2 = results_dict[row[1]][row[3]]['notewise_On_50'][-1]
#         f1_syst1 += [f_syst1]
#         f1_syst2 += [f_syst2]
#     pairs_f1_am += [["{0:.1f}".format(100*np.mean(f1_syst1)),"{0:.1f}".format(100*np.mean(f1_syst2))]]


#### Plot choices with error bars

# dict_stats_bootstrap_cl = {}
# for pair in pairs:
#     data = answers_cl[np.logical_and(answers_cl[:,2]==pair[0], answers_cl[:,3]==pair[1])]
#     result_bootstrap = bootstrap(pairwise_comparison,data,with_majority=False,with_difficulty=False)
#     dict_stats_bootstrap_cl[str(pair)] = result_bootstrap
#
# dict_stats_bootstrap_am = {}
# for pair in pairs:
#     data = answers_am[np.logical_and(answers_am[:,2]==pair[0], answers_am[:,3]==pair[1])]
#     result_bootstrap = bootstrap(pairwise_comparison,data,with_majority=False,with_difficulty=False)
#     dict_stats_bootstrap_am[str(pair)] = result_bootstrap
#
# choices_cl = []
# stds_cl = []
# for pair in pairs:
#     stats = dict_stats_bootstrap_cl[str(pair)]
#     choices_cl += [[stats[0][0],1-stats[0][0]]]
#     stds_cl += [[stats[1][0]]]
# choices_cl = np.array(choices_cl)
# stds_cl = np.array(stds_cl)
#
# choices_am = []
# stds_am = []
# for pair in pairs:
#     stats = dict_stats_bootstrap_am[str(pair)]
#     choices_am += [[stats[0][0],1-stats[0][0]]]
#     stds_am += [[stats[1][0]]]
# choices_am = np.array(choices_am)
# stds_am = np.array(stds_am)
#
# fig, (ax0,ax1) = plt.subplots(1,2)
#
#
# ax0.barh(r, 1-choices_cl[:,0], xerr=stds_cl,capsize=2, color='tab:blue', edgecolor='black', height=barWidth)
# ax0.barh(r, 1-choices_cl[:,1], left=1-choices_cl[:,0], color='tab:red', edgecolor='black', height=barWidth)
# for i in range(len(pairs)):
#     ax0.text(-0.05,i+0.15,pairs[i][0],ha='right', va='center')
#     ax0.text(-0.05,i-0.15,'F1: '+pairs_f1_cl[i][0],ha='right', va='center',fontsize=9)
#     ax0.text(1.05,i+0.15,pairs[i][1],ha='left', va='center')
#     ax0.text(1.05,i-0.15,'F1: '+pairs_f1_cl[i][1],ha='left', va='center',fontsize=9)
#
# ax0.axes.yaxis.set_visible(False)
# ax0.grid(color='grey', linestyle='-', linewidth=1,axis='x')
# ax0.set_axisbelow(True)
# ax0.set_frame_on(False)
# ax0.set_title('ENSTDkCl')
#
# ax1.barh(r, 1-choices_am[:,0], xerr=stds_am,capsize=2, color='tab:blue', edgecolor='black', height=barWidth)
# ax1.barh(r, 1-choices_am[:,1], left=1-choices_am[:,0], color='tab:red', edgecolor='black', height=barWidth)
# for i in range(len(pairs)):
#     ax1.text(-0.05,i+0.15,pairs[i][0],ha='right', va='center')
#     ax1.text(-0.05,i-0.15,'F1: '+pairs_f1_am[i][0],ha='right', va='center',fontsize=9)
#     ax1.text(1.05,i+0.15,pairs[i][1],ha='left', va='center')
#     ax1.text(1.05,i-0.15,'F1: '+pairs_f1_am[i][1],ha='left', va='center',fontsize=9)
#
# ax1.axes.yaxis.set_visible(False)
# ax1.grid(color='grey', linestyle='-', linewidth=1,axis='x')
# ax1.set_axisbelow(True)
# ax1.set_frame_on(False)
# ax1.set_title('ENSTDkAm')
# plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=None)
#
# fig.suptitle('Proportion of choices among all answers for two different recording conditions')
#
# plt.show()
