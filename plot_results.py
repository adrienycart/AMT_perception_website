import os
import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import codecs


import warnings
warnings.filterwarnings("error")





# ############################################
# ##### USERS
# ############################################








filecp = codecs.open('db_csv/user_data.csv', encoding = 'utf-8')
users = np.genfromtxt(filecp,dtype=object,delimiter=";")
users = users[1:,:]
n_users = users.shape[0]
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

####       0            1         2          3        4         5        6            7           8        9           10
#### ['question_id' 'example' 'system1' 'system2' 'user_id' 'answer' 'recognised' 'difficulty' 'time'  'F_syst1' , 'F_syst2']


results_dict = {}
feature_dir = 'precomputed_features'

systems = ['cheng','google',"kelz","lisu"]
pairs = []
for i in range(len(systems)):
    for j in range(i+1,len(systems)):
        pairs += [[systems[i],systems[j]]]

for example in np.unique(answers[:,1]):
    example_dir = os.path.join(feature_dir,example)
    results_dict[example] = {}
    for system in systems:
        results = pickle.load(open(os.path.join(example_dir,system+'.pkl'), "rb"))
        results_dict[example][system]=results

### Pairwise comparisons


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

r = np.array(list(range(len(pairs))))


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

print pairs_f1

#### Plot choices
#
# plt.barh(r, 1-choices[:,0], color='tab:blue', edgecolor='black', height=barWidth)
# plt.barh(r, 1-choices[:,1], left=1-choices[:,0], color='tab:red', edgecolor='black', height=barWidth)
# for i in range(len(pairs)):
#     plt.text(-0.05,i+0.15,pairs[i][0],ha='right', va='center')
#     plt.text(-0.05,i-0.15,'F1: '+pairs_f1[i][0],ha='right', va='center',fontsize=9)
#     plt.text(1.05,i+0.15,pairs[i][1],ha='left', va='center')
#     plt.text(1.05,i-0.15,'F1: '+pairs_f1[i][1],ha='left', va='center',fontsize=9)
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
# labels_f1 = [" - ".join(pair) for pair in pairs_f1]
# for i in range(len(pairs)):
#     plt.text(-0.05,i+0.15,labels[i],ha='right', va='center')
#     plt.text(-0.05,i-0.15,'F1s: '+labels_f1[i],ha='right', va='center',fontsize=8)
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






dict_stats = {}
for pair in pairs:

    choices = [0,0]
    f1_choices = [0,0,0]
    majorities = [0,0,0]

    f1_agreement_each = 0
    # For each answer
    data = answers[np.logical_and(answers[:,2]==pair[0], answers[:,3]==pair[1])]

    for row in data:
        choice = int(row[5])
        f_syst1 = results_dict[row[1]][row[2]]['notewise_On_50'][-1]
        f_syst2 = results_dict[row[1]][row[3]]['notewise_On_50'][-1]
        f_choice = 1-int(f_syst1>f_syst2)

        # print f_syst1, f_syst2, f_choice, choice
        choices[choice] += 1
        if f_choice == 0:
            f1_choices[0] += 1
        else:
            f1_choices[2] += 1

        if choice == f_choice:
            f1_agreement_each += 1

    # # Majority

    f1_agreement_majority = 0
    total_questions_not_draw = 0
    f1_syst1 = []
    f1_syst2 = []
    for q_id in np.unique(data[:,0]):
        data_q = data[data[:,0]==q_id,:]

        fs_syst1 = results_dict[data_q[0,1]][data_q[0,2]]['notewise_On_50'][-1]
        fs_syst2 = results_dict[data_q[0,1]][data_q[0,3]]['notewise_On_50'][-1]
        f_choice = 1-int(f_syst1>f_syst2)

        vote = np.sum(data_q[:,5].astype(int))

        f1_syst1 += [f_syst1]
        f1_syst2 += [f_syst2]

        if vote < 2:
            total_questions_not_draw+=1
            majorities[0] += 1
            if f_choice == 0:
                f1_agreement_majority+=1
        elif vote == 2:
            majorities[1] += 1
        elif vote > 2:
            majorities[2] += 1
            total_questions_not_draw+=1
            if f_choice == 1:
                f1_agreement_majority+=1

        # print f1_syst1, np.mean(f1_syst1)
        dict_stats[str(pair)] = [f1_agreement_each/float(data.shape[0]),f1_agreement_majority/float(total_questions_not_draw),choices,f1_choices,majorities]

for key in dict_stats.keys():
    print key, dict_stats[key]

f1_agreement_each = []
f1_agreement_majority = []
choices = []
majority = []
f1_choices = []
for pair in pairs:
    stats = dict_stats[str(pair)]
    f1_agreement_each += [[stats[0],1-stats[0]]]
    f1_agreement_majority += [[stats[1],1-stats[1]]]
    choices += [stats[2]]
    f1_choices += [stats[3]]
    majority += [stats[4]]

f1_agreement_each = np.array(f1_agreement_each)
f1_agreement_majority = np.array(f1_agreement_majority)

labels = [" - ".join(pair) for pair in pairs]


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


dict_stats = {}
check_dict = {}
for pair in pairs:



    f1_agreement_each_all = []
    f1_agreement_majority_all = []

    # For each answer
    data = answers[np.logical_and(answers[:,2]==pair[0], answers[:,3]==pair[1])]
    # For each answern, dropping difficulty==5
    # data = data[data[:,7].astype(int)<4]

    for measure in ['framewise','notewise_On_25','notewise_On_50','notewise_On_75','notewise_On_100','notewise_OnOff_25','notewise_OnOff_50','notewise_OnOff_75','notewise_OnOff_100']:
        f1_agreement_each = 0
        for row in data:
            choice = int(row[5])
            f_syst1 = results_dict[row[1]][row[2]][measure][-1]
            f_syst2 = results_dict[row[1]][row[3]][measure][-1]
            f_choice = 1-int(f_syst1>f_syst2)

            if choice == f_choice:
                f1_agreement_each += 1
        f1_agreement_each_all += [[f1_agreement_each,float(data.shape[0])]]

    # # Majority
    check = {}
    for measure in ['framewise','notewise_On_25','notewise_On_50','notewise_On_75','notewise_On_100','notewise_OnOff_25','notewise_OnOff_50','notewise_OnOff_75','notewise_OnOff_100']:
        check[measure] = []
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
                majorities[0] += 1
                if f_choice == 0:
                    f1_agreement_majority+=1
            elif vote == 2:
                pass
            elif vote > 2:
                majorities[2] += 1
                total_questions_not_draw+=1
                if f_choice == 1:
                    f1_agreement_majority+=1

        f1_agreement_majority_all += [[f1_agreement_majority,float(total_questions_not_draw)]]

    dict_stats[str(pair)] = [np.array(f1_agreement_each_all),np.array(f1_agreement_majority_all)]


f1_agreement_each=[]
f1_agreement_majority=[]
for pair in pairs:
    f1_agreement_each += [dict_stats[str(pair)][0][:,0]/dict_stats[str(pair)][0][:,1]]
    f1_agreement_majority += [dict_stats[str(pair)][1][:,0]/dict_stats[str(pair)][1][:,1]]
f1_agreement_each = np.array(f1_agreement_each)
f1_agreement_majority = np.array(f1_agreement_majority)


# ######
# ## ON-NOTEWISE ONLY
# f1_agreement_each = f1_agreement_each[:,1:5]
# f1_agreement_majority = f1_agreement_majority[:,1:5]
# bar_labels = ['25ms','50ms','75ms','100ms']
# n_bars = f1_agreement_each.shape[1]
# colors = [np.array([1.0,1,1])-(i+1)/float(n_bars)*np.array([0,1,1])for i in range(n_bars)]

######
## ON-NOTEWISE and FRAMEWISE
f1_agreement_each = f1_agreement_each[:,:5]
f1_agreement_majority = f1_agreement_majority[:,:5]
bar_labels = ['Frame','On\n25ms','On\n50ms','On\n75ms','On\n100ms']
n_bars = f1_agreement_each.shape[1]
colors = ['tab:green']+[np.array([1.0,1,1])-(i+1)/float(n_bars)*np.array([0,1,1])for i in range(4)]


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



single_barWidth = barWidth/n_bars


# # ### Plot for all answers

# for i in range(n_bars):
#     plt.barh(r-single_barWidth/2+(i-1)*single_barWidth, f1_agreement_each[:,i],
#         color=colors[i], edgecolor='black',
#         height=single_barWidth,label=bar_labels[i])
#
# frame1 = plt.gca()
# plt.yticks(r,labels)
# plt.grid(color='grey', linestyle='-', linewidth=1,axis='x')
# frame1.set_axisbelow(True)
# plt.box(False)
# plt.xlim((0,1.1))
# plt.title('Agreement between raters and F-measure,\n with various onset thresholds (all answers)')
# plt.tight_layout(rect=[0, 0, 1, 1])
# plt.legend()
# plt.show()
#
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

#### ACROSS ALL SYSTEM PAIRS

# f1_agreement_each=[]
# f1_agreement_majority=[]
# for pair in pairs:
#     f1_agreement_each += [[dict_stats[str(pair)][0][:,0],dict_stats[str(pair)][0][:,1]]]
#     f1_agreement_majority += [[dict_stats[str(pair)][1][:,0],dict_stats[str(pair)][1][:,1]]]
# f1_agreement_each = np.array(f1_agreement_each)
# f1_agreement_majority = np.array(f1_agreement_majority)
#
# f1_agreement_each = np.sum(f1_agreement_each[:,0,:],axis=0)/np.sum(f1_agreement_each[:,1,:],axis=0).astype(float)
# f1_agreement_majority = np.sum(f1_agreement_majority[:,0,:],axis=0)/np.sum(f1_agreement_majority[:,1,:],axis=0).astype(float)
#
# labels = ["Frame","On\n25ms","On\n50ms","On\n75ms","On\n100ms","OnOff\n25ms","OnOff\n50ms","OnOff\n75ms","OnOff\n100ms"]
# colors = ['tab:green']+[np.array([1.0,1,1])-(i+1)/float(4)*np.array([0,1,1]) for i in range(4)]+[np.array([1.0,1,1])-(i+1)/float(4)*np.array([1,1,0]) for i in range(4)]
#
# plt.bar(list(range(len(f1_agreement_each))), f1_agreement_each, color=colors, edgecolor='black', width=barWidth)
# frame1 = plt.gca()
# plt.xticks(list(range(len(f1_agreement_each))),labels)
# plt.grid(color='grey', linestyle='-', linewidth=1,axis='y')
# frame1.set_axisbelow(True)
# plt.box(False)
# plt.ylim((0,1.05))
# plt.title('Agreement between raters and various F-measures\n(all answers)')
# plt.tight_layout(rect=[0, 0, 1, 1])
# plt.show()

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
print F_measures
answers = np.concatenate([answers,F_measures],axis=1)
print answers

#### F-measure difference vs disagreement

# F_mes_diffs = []
# agreements = []
# for q_id in np.unique(answers[:,0]):
#     data_q = answers[answers[:,0]==q_id,:]
#     F_mes_diffs += [abs(float(data_q[0,-1])-float(data_q[0,-2]))]
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
# plt.title("Difference in F-measure vs. inter-rater agreement")
# plt.show()
#
# #### F-measure difference vs reported difficulty
#
# F_mes_diffs = np.abs(answers[:,-1]-answers[:,-2])
# difficulties = answers[:,7]
#
# data = [F_mes_diffs[difficulties==str(i)].astype(float) for i in range(1,6)]
# print data
# plt.violinplot(data, positions=range(1,6), vert=True, widths=0.3,
#                        showextrema=True, showmedians=True)
# plt.xticks(range(1,6),range(1,6))
# plt.title("Difference in F-measure vs. reported difficulty")
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
# plt.violinplot(data, positions=range(1,6), vert=True, widths=0.3,
#                        showextrema=True, showmedians=True)
# plt.xticks(range(1,6),range(1,6))
# plt.title("F-measure of chosen option vs. reported difficulty")
# plt.show()


#### Agreement with F-measure vs difficulty
# f_choice = 1-int(f_syst1>f_syst2)
# f_choices = 1-(answers[:,-2]>answers[:,-1]).astype(int)
# agree = (f_choices==answers[:,5].astype(int)).astype(int)
# difficulties = answers[:,7]
# data = [np.mean(agree[difficulties==str(i)]) for i in range(1,6)]
#
# plt.bar([1,2,3,4,5],data,width=barWidth,edgecolor='black')
# frame1 = plt.gca()
# plt.xticks([1,2,3,4,5],[1,2,3,4,5])
# plt.grid(color='grey', linestyle='-', linewidth=1,axis='y')
# frame1.set_axisbelow(True)
# plt.box(False)
# plt.ylim((0,1.05))
# plt.title('Agreement between raters and F-measure\nfor each reported difficulty')
# plt.tight_layout(rect=[0, 0, 1, 1])
# plt.show()

#### Known vs Unknown pieces

known = answers[:,6]=='True'
unknown = answers[:,6]=='False'

agree_known = 0
agree_unknown = 0
difficulty_known = [0,0,0,0,0]
difficulty_unknown = [0,0,0,0,0]
for row in answers:
    f_choice = 1-int(row[-2]>row[-1])
    agree = f_choice == int(row[5])

    if row[6] == 'True':
        difficulty_known[int(row[7])-1] += 1
        if agree:
            agree_known += 1
    elif row[6] == 'False':
        difficulty_unknown[int(row[7])-1] += 1
        if agree:
            agree_unknown += 1
difficulty_known = np.array(difficulty_known)/float(sum(difficulty_known))
difficulty_unknown = np.array(difficulty_unknown)/float(sum(difficulty_unknown))


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

difficulties = np.concatenate([difficulty_known[None,:],difficulty_unknown[None,:]],axis=0)
print difficulties.shape

start = np.zeros([2])
for i in range(5):
    end = start + difficulties[:,i]
    plt.barh([0,1], difficulties[:,i], left=np.sum(difficulties[:,:i],axis=1), color=np.array([1.0,1,1])-(i+1)/5.0*np.array([0,1,1]), edgecolor='black', height=barWidth)

frame1 = plt.gca()
plt.yticks([0,1],['Known','Unknown'])
# plt.grid(color='grey', linestyle='-', linewidth=1,axis='y')
frame1.set_axisbelow(True)
plt.box(False)
# plt.ylim((0,1.05))
plt.title('Reported difficulty for known and unknown examples')
# plt.tight_layout(rect=[0, 0, 1, 1])
plt.show()


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

goldmsi = []
avg_difficulty = []
avg_agreement = []
known = []
for user in users:
    data = answers[answers[:,4]==user[0]]
    goldmsi += [float(user[5])]
    avg_difficulty += [np.mean(data[:,7].astype(int))]

    f_choice = 1-(data[:,-2]>data[:,-1]).astype(int)
    agree = f_choice == data[:,5].astype(int)
    avg_agreement += [np.mean(agree.astype(int))]

    known += [np.mean((data[:,6]=='True').astype(int))]


plt.scatter(goldmsi,avg_difficulty)
frame1 = plt.gca()
plt.xticks(range(2,8),range(2,8))
plt.yticks(range(6),range(6))
plt.xlabel('GoldMSI Score')
plt.ylabel('Average reported difficulty')
plt.grid(color='grey', linestyle='-', linewidth=1,axis='y')
frame1.set_axisbelow(True)
plt.box(False)

plt.title('Average reported difficulty vs. GoldMSI Score')
plt.tight_layout(rect=[0, 0, 1, 1])
plt.show()

# plt.scatter(goldmsi,known)
# frame1 = plt.gca()
# plt.xticks(range(2,8),range(2,8))
# plt.yticks(range(6),range(6))
# plt.xlabel('GoldMSI Score')
# plt.ylabel('Average reported difficulty')
# plt.grid(color='grey', linestyle='-', linewidth=1,axis='y')
# frame1.set_axisbelow(True)
# plt.box(False)
#
# plt.title('Average reported difficulty vs. GoldMSI Score')
# plt.tight_layout(rect=[0, 0, 1, 1])
# plt.show()
