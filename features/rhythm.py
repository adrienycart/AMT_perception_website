import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

SMALL_VALUE = 0.0001


# TESTED
def rhythm_histogram(intervals_output, intervals_target):
    # return the logged spectral flatness ratio of the IOIs
    # 1. spectral flatness for IOI of the output transcription
    # 2. spectral flatness for IOI of the ground truth music piece
    ioi_output = [float(intervals_output[idx][1] - intervals_output[idx][0]) for idx in range(len(intervals_output))]
    ioi_target = [float(intervals_target[idx][1] - intervals_target[idx][0]) for idx in range(len(intervals_target))]

    # generate bins
    bins = [i*0.01 for i in range(10)]
    bins += [0.1+i*0.1 for i in range(20)]

    histogram_output = np.histogram(ioi_output, bins=bins)[0]
    histogram_target = np.histogram(ioi_target, bins=bins)[0]

    histogram_output = [max(SMALL_VALUE, histogram_output[idx]) for idx in range(len(histogram_output))]
    histogram_target = [max(SMALL_VALUE, histogram_target[idx]) for idx in range(len(histogram_target))]

    log_gmean_output = np.mean(np.log(histogram_output))
    log_gmean_target = np.mean(np.log(histogram_target))
    mean_output = np.mean(histogram_output)
    mean_target = np.mean(histogram_target)

    return log_gmean_output - np.log(mean_output), log_gmean_target - np.log(mean_target)


# TESTED: current
# TODO: add standard diviation for each cluster
def rhythm_dispersion(intervals_output, intervals_target):
    # return changes in k-means clusters
    # 1. change in standard deviations (k-means doesn't have std definition...)
    # 2. center drift (average drift and max drift)

    ioi_output = [float(intervals_output[idx][1] - intervals_output[idx][0]) for idx in range(len(intervals_output))]
    ioi_target = [float(intervals_target[idx][1] - intervals_target[idx][0]) for idx in range(len(intervals_target))]

    # initialise cluster
    bins = [i*0.01 for i in range(10)]
    bins += [0.1+i*0.1 for i in range(20)]
    histogram_target = np.histogram(ioi_target, bins=bins)[0]
    means = []
    for i in range(1, len(histogram_target)-1):
        if histogram_target[i] > 0 and histogram_target[i] >= histogram_target[i-1] and histogram_target[i] >= histogram_target[i+1]:
            means.append(np.mean([bins[i], bins[i+1]]))
            
    if len(means) == 0:
        return 0.0, 0.0

    # k-means on target intervals
    moving = 1
    while moving > 0.0001:
        # cluster
        ioi_target_labels = [np.argmin(abs(np.array(means) - ioi_target[idx])) for idx in range(len(ioi_target))]
        # calculate new means
        new_means = []
        for label in range(len(means)):
            indexs = np.array([i for i in range(len(ioi_target)) if ioi_target_labels[i] == label])
            if len(indexs) > 0:
                new_means.append(np.mean(np.array(ioi_target)[indexs]))
            else:
                new_means.append(means[label])
        # calculate centre moving
        moving = np.sum(abs(np.array(new_means) - np.array(means)))
        # update means
        means = new_means
        # print(means)

    # print('--')
    # k-means on output intervals
    means_output = [means[i] for i in range(len(means))] # copy target means
    moving = 1
    while moving > 0.0001:
        # cluster
        ioi_output_labels = [np.argmin(abs(np.array(means_output) - ioi_output[idx])) for idx in range(len(ioi_output))]
        # calculate new means
        new_means_output = []
        for label in range(len(means_output)):
            indexs = np.array([i for i in range(len(ioi_output)) if ioi_output_labels[i] == label])
            if len(indexs) > 0:
                new_means_output.append(np.mean(np.array(ioi_output)[indexs]))
            else:
                new_means_output.append(means_output[label])
        # calculate centre moving
        moving = np.sum(abs(np.array(new_means_output) - np.array(means_output)))
        # update means
        means_output = new_means_output
        # print(means_output)

    # cluster centre drift
    drifts = [abs(means[idx] - means_output[idx]) for idx in range(len(means))]

    return np.mean(drifts), max(drifts)