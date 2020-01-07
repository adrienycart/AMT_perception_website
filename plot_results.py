import os
import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import codecs

# ############################################
# ##### USERS
# ############################################
#
#
# filecp = codecs.open('db_csv/user_data.csv', encoding = 'utf-8')
# users = np.genfromtxt(filecp,dtype=object,delimiter=";")
# users = users[1:,:]
# n_users = users.shape[0]
# #### 0         1      2     3    4          5
# #### user;n_answers;gender;age;disability;gold_msi_avg
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
# print "n_male",n_male,"n_female",n_female,"n_other",n_other
#
# values = [n_male,n_female,n_other]
# plt.pie(values,labels=['Male',"Female","Non-binary"],autopct=lambda p : '{:.2f}%  ({:,.0f})'.format(p,p * sum(values)/100),colors=['tab:blue','tab:red','tab:green'])
# plt.axis('equal')
# plt.title('Gender distribution')
# plt.show()
#
# ### Age distribution
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
#
#
# ### GoldMSI distribution
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
# ### Number of answers distrbution
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
#

############################################
##### PAIRWISE RESPONSES
############################################

filecp = codecs.open('db_csv/answers_data.csv', encoding = 'utf-8')
answers = np.genfromtxt(filecp,dtype=object,delimiter=";")
answers = answers[1:,:]
# n_users = answers.shape[0]

####       0            1         2          3        4         5        6            7           8
#### ['question_id' 'example' 'system1' 'system2' 'user_id' 'answer' 'recognised' 'difficulty' 'time']


### Pairwise comparisons
systems = ['cheng','google',"kelz","lisu"]
pairs = []
for i in range(len(systems)):
    for j in range(i+1,len(systems)):
        pairs += [[systems[i],systems[j]]]

dict_stats = {}

for pair in pairs:
    data = answers[np.logical_and(answers[:,2]==pair[0], answers[:,3]==pair[1])]

    avg_choice = np.mean(data[:,5].astype(int))
    difficulties = [np.sum((data[:,7]==str(i)).astype(int)) for i in range(1,6)]
    avg_difficulty = np.mean(data[:,7].astype(int))
    avg_time = np.mean(data[:,8].astype(float))

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

    dict_stats[str(pair)] = [avg_choice,avg_difficulty,avg_time,majority,difficulties]

for key in dict_stats.keys():
    print key, dict_stats[key]

r = list(range(len(pairs)))


choices = []
difficulty = []
majority = []
for pair in pairs:
    stats = dict_stats[str(pair)]
    choices += [[stats[0],1-stats[0]]]
    difficulty += [stats[4]]
    majority += [stats[3]]

choices = np.array(choices)
difficulty = np.array(difficulty)
majority = np.array(majority)

barWidth = 0.75

left_labels = [pair[0] for pair in pairs]
right_labels = [pair[1] for pair in pairs]


#### Plot choices

# plt.barh(r, 1-choices[:,0], color='tab:blue', edgecolor='black', height=barWidth)
# plt.barh(r, 1-choices[:,1], left=1-choices[:,0], color='tab:red', edgecolor='black', height=barWidth)
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
# plt.title('Proportion of choices among all answers')
#
# plt.show()
#
#
# #### Plot difficulties
#
# normalized_difficulty= difficulty/np.sum(difficulty,axis=1).astype(float)[:,None]
# start = np.zeros_like(normalized_difficulty[:,0])
# for i in range(5):
#     end = start + normalized_difficulty[:,i]
#     plt.barh(r, normalized_difficulty[:,i], left=np.sum(normalized_difficulty[:,:i],axis=1), color=np.array([1.0,1,1])-(i+1)/5.0*np.array([0,1,1]), edgecolor='black', height=barWidth)
#
# labels = [" - ".join(pair) for pair in pairs]
# for i in range(len(pairs)):
#     plt.text(-0.05,i,labels[i],ha='right', va='center')
#
# frame1 = plt.gca()
# plt.grid(color='grey', linestyle='-', linewidth=1,axis='x')
# frame1.axes.yaxis.set_visible(False)
# frame1.set_axisbelow(True)
# plt.box(False)
# plt.tight_layout(rect=[0.2, 0, 1, 0.95])
#
# plt.title('Proportion of difficulty ratings', x=0.4)
#
# plt.show()

#### Plot majority

# normalized_majority= majority/np.sum(majority,axis=1).astype(float)[:,None]
# start = np.zeros_like(normalized_majority[:,0])
# colors = ['tab:blue','grey','tab:red']
# color_labels = ['1st best','Draw','2nd best']
# for i in range(3):
#     end = start + normalized_majority[:,i]
#     plt.barh(r, normalized_majority[:,i], left=np.sum(normalized_majority[:,:i],axis=1), color=colors[i], edgecolor='black', height=barWidth,label=color_labels[i])
#
# for i in range(len(pairs)):
#     plt.text(-0.05,i,pairs[i][0],ha='right', va='center')
#     plt.text(1.05,i,pairs[i][1],ha='left', va='center')
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
##### AGREEMENT F-MEASURE - ANSWERS
############################################

results_dict = {}
feature_dir = 'precomputed_features'

for example in np.unique(answers[:,1]):
    example_dir = os.path.join(feature_dir,example)
    results_dict[example] = {}
    for system in systems:
        results = pickle.load(open(os.path.join(example_dir,system+'.pkl'), "rb"))
        results_dict[example][system]=results



f1_agreement_majority = 0
f1_diff_equal = 0

dict_stats = {}

for pair in pairs:
    f1_agreement_each = 0
    # For each answer
    data = answers[np.logical_and(answers[:,2]==pair[0], answers[:,3]==pair[1])]

    for row in data:
        choice = int(row[5])
        f_syst1 = results_dict[row[1]][row[2]]['notewise_On_50'][-1]
        f_syst2 = results_dict[row[1]][row[3]]['notewise_On_50'][-1]
        f_choice = 1-int(f_syst1>f_syst2)

        # print f_syst1, f_syst2, f_choice, choice

        if choice == f_choice:
            f1_agreement_each += 1

    # # Majority
    f1_agreement_majority = 0
    total_questions_not_draw = 0
    f1_diff_equal = []
    for q_id in np.unique(data[:,0]):
        data_q = data[data[:,0]==q_id,:]

        f_syst1 = results_dict[data_q[0,1]][data_q[0,2]]['notewise_On_50'][-1]
        f_syst2 = results_dict[data_q[0,1]][data_q[0,3]]['notewise_On_50'][-1]
        f_choice = 1-int(f_syst1>f_syst2)

        vote = np.sum(data_q[:,5].astype(int))

        if vote < 2:
            total_questions_not_draw+=1
            if f_choice == 0:
                f1_agreement_majority+=1
        elif vote == 2:
            f1_diff_equal += [abs(f_syst1-f_syst2)]
        elif vote > 2:
            total_questions_not_draw+=1
            if f_choice == 1:
                f1_agreement_majority+=1

    dict_stats[str(pair)] = [f1_agreement_each/float(data.shape[0]),f1_agreement_majority/float(total_questions_not_draw),np.mean(f1_diff_equal)]

choices = []
majority = []
for pair in pairs:
    stats = dict_stats[str(pair)]
    choices += [[stats[0],1-stats[0]]]
    majority += [[stats[1],1-stats[1]]]

choices = np.array(choices)
majority = np.array(majority)

labels = [" - ".join(pair) for pair in pairs]

### Plot for all answers
plt.bar(r, choices[:,0], color='tab:blue', edgecolor='black', width=barWidth)

frame1 = plt.gca()
plt.xticks(r,labels)
plt.grid(color='grey', linestyle='-', linewidth=1,axis='y')
frame1.set_axisbelow(True)
plt.box(False)

plt.title('Agreement between raters and F-measure (all answers)')

plt.show()

### Plot for majority voting only
plt.bar(r, majority[:,0], color='tab:blue', edgecolor='black', width=barWidth)

frame1 = plt.gca()
plt.xticks(r,labels)
plt.grid(color='grey', linestyle='-', linewidth=1,axis='y')
frame1.set_axisbelow(True)
plt.box(False)

plt.title('Agreement between raters and F-measure (majority voting, no draws)')

plt.show()
