import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import ttest_1samp, ttest_ind
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
import pandas as pd
from matplotlib import rc

plt.rcParams.update({'font.size': 13.5,'font.family' : 'serif'})
rc('text', usetex=True)

# base_folder = "results_metric"
# base_folder = "results_metric/10_folds"
base_folder = "results_metric/20_folds_maxdiff4_cons_nozero"
n_folds = 20

folders_to_compare = [
                        ["all_features","All"],
                        # ["all_features_maxdiffic_2","All (Difficuly<5)"],
                        # ["all_features_maxdiffic_4","All (Difficuly<3)"],
                        ["no_benchmark",'NoBench'],
                        ["only_benchmark",'BenchOnly'],
                        ["no_high_low",'NoHighLow'],
                        ["no_loud",'NoLoud'],
                        ["no_out_key",'NoOutKey'],
                        ["no_specific",'NoSpecific'],
                        ["no_poly",'NoPoly'],
                        ["no_repeat",'NoRepeat'],
                        ["no_rhythm",'NoRhythm'],
                        ["no_consonance",'NoConsonance'],
                        # ["no_consonance_diff",'NoConsDiff'],
                        ["no_specific_consonance",'NoSpecCons'],
                        ["no_framewise",'NoFramewise'],
                        # ["no_useless",'NoUseless'],

                        ]

folders_to_compare = [[os.path.join(base_folder,folder),name] for folder,name in folders_to_compare]


reference_matrix = []
reference_matrix_conf = []
reference_matrix_agg = []
file_handle = open(os.path.join(base_folder,"all_features",'all_folds.pkl'), "rb")
reference_results = pickle.load(file_handle)

results_F1=[]
results_F1_conf=[]
results_F1_agg =[]

for i in range(n_folds):
    reference_matrix += [reference_results['fold'+str(i)]['repeat_agreement']]
    reference_matrix_conf += [reference_results['fold'+str(i)]['repeat_agreement_conf']]
    reference_matrix_agg += [reference_results['fold'+str(i)]['repeat_agreement_agg']]
    results_F1 += [reference_results['fold'+str(i)]['agreement_F1']]
    results_F1_conf += [reference_results['fold'+str(i)]['agreement_F1_conf']]
    results_F1_agg += [reference_results['fold'+str(i)]['agreement_F1_agg']]


p_values_all = []

for folder,name in folders_to_compare:
    file_handle = open(os.path.join(folder,'all_folds.pkl'), "rb")
    all_results = pickle.load(file_handle)
    t_values_f = []
    t_values_all_features = []
    t_values_f_conf = []
    t_values_all_features_conf = []
    t_values_f_agg = []
    t_values_all_features_agg = []
    for i in range(n_folds):
        agreement = all_results['fold'+str(i)]['repeat_agreement']
        agreement_conf = all_results['fold'+str(i)]['repeat_agreement_conf']
        agreement_agg = all_results['fold'+str(i)]['repeat_agreement_agg']

        t_values_f += [ttest_1samp(agreement,results_F1[i])[0]]
        t_values_f_conf += [ttest_1samp(agreement,results_F1_conf[i])[0]]
        t_values_f_agg += [ttest_1samp(agreement,results_F1_agg[i])[0]]

        t_values_all_features += [ttest_ind(agreement,reference_matrix[i],equal_var=False)[0]]
        t_values_all_features_conf += [ttest_ind(agreement_conf,reference_matrix_conf[i],equal_var=False)[0]]
        t_values_all_features_agg += [ttest_ind(agreement_agg,reference_matrix_agg[i],equal_var=False)[0]]

    print '------------------'
    print name

    p_value_f = ttest_1samp(t_values_f,0)[1]
    p_value_f_conf = ttest_1samp(t_values_f_conf,0)[1]
    p_value_f_agg = ttest_1samp(t_values_f_agg,0)[1]

    p_value_all_features = ttest_1samp(t_values_all_features,0)[1]
    p_value_all_features_conf = ttest_1samp(t_values_all_features_conf,0)[1]
    p_value_all_features_agg = ttest_1samp(t_values_all_features_agg,0)[1]

    p_values_all += [p_value_all_features_conf]

    print t_values_all_features_conf
    print "p_value_f", p_value_f
    print "p_value_f_conf", p_value_f_conf
    print "p_value_f_agg", p_value_f_agg
    print
    print "p_value_all_features", p_value_all_features
    print "p_value_all_features_conf", p_value_all_features_conf
    print "p_value_all_features_agg", p_value_all_features_agg


##### PLOT RESULTS
print 'baseline', np.round(np.mean(results_F1),3), np.round(np.mean(results_F1_conf),3)
results_avg = []
results_avg_conf = []
for i,(folder,name) in enumerate(folders_to_compare):
    file_handle = open(os.path.join(folder,'all_folds.pkl'), "rb")
    all_results = pickle.load(file_handle)
    results_concat = []
    results_concat_conf = []
    for fold in range(n_folds):
        results_concat_conf += all_results['fold'+str(fold)]['repeat_agreement_conf']
        results_concat += all_results['fold'+str(fold)]['repeat_agreement']

    results_avg += [np.mean(results_concat)]
    results_avg_conf += [np.mean(results_concat_conf)]


sort_idx = np.argsort(results_avg_conf,)
for i,idx in enumerate(sort_idx):
    if folders_to_compare[idx][1] == 'All':
        color = 'black'
    else:
        color = 'tab:blue'
    plt.barh(i,results_avg_conf[idx],color=color,edgecolor='black')
    significance = ''
    if p_values_all[idx] < 0.1:
        significance = '*'
    if p_values_all[idx] < 0.05:
        significance = '**'
    if p_values_all[idx] < 0.01:
        significance = '***'
    # if p_values_all[idx] < 0.01:
    #     significance = '***'
    plt.text(results_avg_conf[idx]+0.001,i,significance,verticalalignment='center')

    print os.path.basename(folders_to_compare[i][1]), np.round(results_avg[i]*100,1),"&", np.round(results_avg_conf[i]*100,1)
    # if 'all_features' == os.path.basename(folder):
    #     plt.plot([mean,mean],[0,len(folders_to_compare)],linestyle='--',color='grey')

plt.yticks(range(len(folders_to_compare)),[folders_to_compare[sort_idx[i]][1] for i in range(len(folders_to_compare))])
plt.plot([np.mean(results_F1_conf),np.mean(results_F1_conf)],[-0.5,len(folders_to_compare)-0.5],linestyle='--',color='black')
plt.xlim([0.86,0.895])
plt.xlabel(r'$A_{conf}$')
plt.tight_layout()
plt.show()
