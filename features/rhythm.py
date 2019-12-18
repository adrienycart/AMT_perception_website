import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

SMALL_VALUE = 0.5

def rhythm_histogram(intervals_output, intervals_target):
	# return the spectral flatness ratio of the IOIs
	# 1. spectral flatness for IOI of the output transcription
	# 2. spectral flatness for IOI of the ground truth music piece
	ioi_output = [intervals_output[idx][1] - intervals_output[idx][0] for idx in range(len(intervals_output))]
	ioi_target = [intervals_target[idx][1] - intervals_target[idx][0] for idx in range(len(intervals_target))]

	# generate bins
	bins = [i*0.01 for i in range(10)]
	bins += [0.1+i*0.1 for i in range(20)]

	histogram_output = np.histogram(ioi_output, bins=bins)[0]
	histogram_target = np.histogram(ioi_target, bins=bins)[0]

	for i in range(len(bins)-1):
		if histogram_output[i] == 0:
			histogram_output[i] = SMALL_VALUE
		if histogram_target[i] == 0:
			histogram_target[i] = SMALL_VALUE

	gmean_output = stats.mstats.gmean(histogram_output)
	gmean_target = stats.mstats.gmean(histogram_target)
	mean_output = np.mean(histogram_output)
	mean_target = np.mean(histogram_target)
	print(gmean_target)

	return gmean_output / mean_output, gmean_target / mean_target


