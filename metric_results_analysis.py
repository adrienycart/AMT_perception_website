import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import ttest_rel

folders_to_compare = [
                        "results_metric/all_features",
                        "results_metric/no_benchmark",
                        "results_metric/only_benchmark",
                        "results_metric/only_high_low",
                        "results_metric/only_loud",
                        "results_metric/only_out_key",
                        "results_metric/only_specific_pitch",
                        "results_metric/only_framewise",
                        "results_metric/only_notewise_w_benchmark",
                        "results_metric/only_notewise_wo_benchmark",
                        ]
results_F1=[
0.809278350515,
0.795081967213,
0.798387096774,
0.798387096774,
0.826923076923,
0.814516129032,
0.807692307692,
0.820512820513,
0.79128440367,
0.807692307692,
]

print 'baseline', np.round(np.mean(results_F1),3)
for folder in folders_to_compare:
    file_handle = open(os.path.join(folder,'all_folds.pkl'), "rb")
    all_results = pickle.load(file_handle)
    results_concat = []
    for i in range(10):
        results_concat += all_results['fold'+str(i)]['repeat_agreement']
        print 'fold'+str(i), np.mean(all_results['fold'+str(i)]['repeat_agreement'])

    # print folder
    # print "results", np.mean(results_concat), 'F', np.mean(results_F1)
    print os.path.basename(folder), np.round(np.mean(results_concat),3)


# for i in range(10):
#     for j, folder in enumerate(folders_to_compare):
#         file_handle = open(os.path.join(folder,'all_folds.pkl'), "rb")
#         all_results = pickle.load(file_handle)
#         results = all_results['fold'+str(i)]['repeat_agreement']
#         mean = np.mean(results)
#         std = np.std(results)
#         baseline = results_F1[i]
#
#         plt.bar(j-0.15,mean,width=0.3,yerr=std,capsize=2,color='tab:blue', edgecolor='black')
#         plt.bar(j+0.15,baseline,width=0.3,color='tab:red', edgecolor='black')
#     plt.title('Fold '+str(i))
#     plt.xticks(range(len(folders_to_compare)),[os.path.basename(fold) for fold in folders_to_compare],rotation=90)
#     plt.tight_layout()
#     plt.show()

reference_samples = []
file_handle = open(os.path.join(folder,'all_folds.pkl'), "rb")
reference_results = pickle.load(file_handle)
for i in range(10):
    reference_samples += [reference_results['fold'+str(i)]['repeat_agreement']]
reference_samples = np.array(reference_samples)

for i,folder in enumerate(folders_to_compare):
    samples = []
    file_handle = open(os.path.join(folder,'all_folds.pkl'), "rb")
    all_results = pickle.load(file_handle)
    for i in range(10):
        samples += [all_results['fold'+str(i)]['repeat_agreement']]
    samples = np.array(samples)

    print folder, ttest_rel(reference_samples,samples,axis=1)
