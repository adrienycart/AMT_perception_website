import os
import numpy as np
import matplotlib.pyplot as plt
import pretty_midi as pm
import mir_eval
import features.utils as utils
from features.rhythm import rhythm_histogram, rhythm_dispersion

import warnings
warnings.filterwarnings("ignore")

result_folder = "validate_rhythm_feature_plots_update"
utils.create_folder(result_folder)
MIDI_path = "app/static/data/all_midi_cut"
systems = ["kelz", "lisu", "google", "cheng"]
fs = 100

N_features = 8
sfo, sfd, stdmean, stdmin, stdmax, drmean, drmin, drmax = range(N_features)
N_computes = 5   # calculate quantize_over_original and noisy_over_original
noise_level = [0.1, 0.2, 0.3]
strict_quantize, quantize, noisy1, noisy2, noisy3 = range(N_computes)
N_outputs = len(os.listdir(MIDI_path)) * len(systems)


def plot_hist(x1, x2, x3, x4, title, limits, filename, n_bins=50):

    plt.figure(figsize=(6.4, 8.2))
    plt.subplot(411)
    plt.hist(x1, bins=n_bins, range=limits)
    plt.ylabel("quantize/original")
    plt.title(title)
    plt.subplot(412)
    plt.hist(x2, bins=n_bins, range=limits)
    plt.ylabel("noisy({:.1f})/original".format(noise_level[0]))
    plt.subplot(413)
    plt.hist(x3, bins=n_bins, range=limits)
    plt.ylabel("noisy({:.1f})/original".format(noise_level[1]))
    plt.subplot(414)
    plt.hist(x4, bins=n_bins, range=limits)
    plt.ylabel("noisy({:.1f})/original".format(noise_level[2]))
    plt.savefig(filename)
    plt.show()


def print_line(values, feature_name, feature_index):
    print(feature_name+"\t| {:.3f} \t {:.3f} \t| {:.3f} \t {:.3f} \t| {:.3f} \t {:.3f} \t| {:.3f} \t {:.3f}".format(np.mean(values[feature_index, quantize]), np.std(values[feature_index, quantize]), np.mean(values[feature_index, noisy1]), np.std(values[feature_index, noisy1]), np.mean(values[feature_index, noisy2]), np.std(values[feature_index, noisy2]), np.mean(values[feature_index, noisy3]), np.std(values[feature_index, noisy3])))

if os.path.exists(result_folder+"/data.npy"):
    values = np.load(result_folder+"/data.npy")
else:
    values = np.zeros((N_features, N_computes, N_outputs), dtype=float)

    idx = 0
    for example in os.listdir(MIDI_path)[:]:
        example_path = os.path.join(MIDI_path, example)
        print((idx / len(systems), example_path))

        # get quantized quarter times
        target_data = pm.PrettyMIDI(os.path.join(example_path, "target.mid"))
        target_PPQ = target_data.resolution
        end_tick = target_data.time_to_tick(target_data.get_end_time())
        ticks = np.arange(0, end_tick, target_PPQ/4)
        quarter_times = np.array([target_data.tick_to_time(t) for t in ticks])

        # # get strict quantized intervals
        # tempo = (quarter_times[-1] - quarter_times[0]) / (len(quarter_times) - 1)
        # quarter_times_strict = np.array([quarter_times[0] + i * tempo for i in range(len(quarter_times))])
        # print(np.sum(np.abs(quarter_times - quarter_times_strict)))

        for system in systems:
            system_data = pm.PrettyMIDI(os.path.join(example_path, system+".mid"))
            notes_system, intervals_system = utils.get_notes_intervals(system_data)

            # get quantized intervals
            intervals_system_quantized = intervals_system.copy()
            for i in range(len(intervals_system)):
                intervals_system_quantized[i][0] = quarter_times[np.argmin(np.abs(quarter_times - intervals_system[i][0]))]

            # check quantization
            # plt.figure()
            # plt.plot([x[0] for x in intervals_system_quantized])
            # plt.plot([x[0] for x in intervals_system])
            # plt.show()

            values[sfo, quantize, idx], values[sfd, quantize, idx] = rhythm_histogram(intervals_system_quantized, intervals_system)
            values[sfo, noisy1, idx], values[sfd, noisy1, idx] = rhythm_histogram(intervals_system, intervals_system, noise=noise_level[0])
            values[sfo, noisy2, idx], values[sfd, noisy2, idx] = rhythm_histogram(intervals_system, intervals_system, noise=noise_level[1])
            values[sfo, noisy3, idx], values[sfd, noisy3, idx] = rhythm_histogram(intervals_system, intervals_system, noise=noise_level[2])

            [values[stdmean, quantize, idx], values[stdmin, quantize, idx], values[stdmax, quantize, idx]], [values[drmean, quantize, idx], values[drmin, quantize, idx], values[drmax, quantize, idx]] = rhythm_dispersion(intervals_system_quantized, intervals_system)
            [values[stdmean, noisy1, idx], values[stdmin, noisy1, idx], values[stdmax, noisy1, idx]], [values[drmean, noisy1, idx], values[drmin, noisy1, idx], values[drmax, noisy1, idx]] = rhythm_dispersion(intervals_system, intervals_system, noise=noise_level[0])
            [values[stdmean, noisy2, idx], values[stdmin, noisy2, idx], values[stdmax, noisy2, idx]], [values[drmean, noisy2, idx], values[drmin, noisy2, idx], values[drmax, noisy2, idx]] = rhythm_dispersion(intervals_system, intervals_system, noise=noise_level[1])
            [values[stdmean, noisy3, idx], values[stdmin, noisy3, idx], values[stdmax, noisy3, idx]], [values[drmean, noisy3, idx], values[drmin, noisy3, idx], values[drmax, noisy3, idx]] = rhythm_dispersion(intervals_system, intervals_system, noise=noise_level[2])

            idx += 1

    np.save(result_folder+"/data.npy", values)

#############################################################
## plot and save distributions
#############################################################

print("plot and save value distributions...")

plot_hist(values[sfo, quantize], values[sfo, noisy1], values[sfo, noisy2], values[sfo, noisy3], limits=(-12, -4), title="spectral flatness of output", filename=result_folder+"/spectral_flatness_output.pdf")
plot_hist(values[sfd, quantize], values[sfd, noisy1], values[sfd, noisy2], values[sfd, noisy3], limits=(-5, 6), title="spectral flatness difference", filename=result_folder+"/spectral_flatness_difference.pdf")
plot_hist(values[stdmean, quantize], values[stdmean, noisy1], values[stdmean, noisy2], values[stdmean, noisy3], limits=(-0.05, 0.2), title="average standard deviation change k-means", filename=result_folder+"/standard_deviation_change_average.pdf")
plot_hist(values[stdmin, quantize], values[stdmin, noisy1], values[stdmin, noisy2], values[stdmin, noisy3], limits=(-0.2, 0.2), title="minimum standard deviation change k-means", filename=result_folder+"/standard_deviation_change_minimum.pdf")
plot_hist(values[stdmax, quantize], values[stdmax, noisy1], values[stdmax, noisy2], values[stdmax, noisy3], limits=(-0.05, 0.25), title="maximum standard deviation change k-means", filename=result_folder+"/standard_deviation_change_maximum.pdf")
plot_hist(values[drmean, quantize], values[drmean, noisy1], values[drmean, noisy2], values[drmean, noisy3], limits=(0, 0.25), title="average drifts k-means", filename=result_folder+"/drifts_average.pdf")
plot_hist(values[drmin, quantize], values[drmin, noisy1], values[drmin, noisy2], values[drmin, noisy3], limits=(0, 0.2), title="minimum drifts k-means", filename=result_folder+"/drifts_minimum.pdf")
plot_hist(values[drmax, quantize], values[drmax, noisy1], values[drmax, noisy2], values[drmax, noisy3], limits=(0, 0.5), title="maximum drifts k-means", filename=result_folder+"/drifts_maximum.pdf")


###############################################################
##   print results
###############################################################

print("-------------------------------------------------------------------------------------------------------------------------------")
print("      over original         \t|     quantize      \t| noisy level: {:.1f}  \t| noisy level: {:.1f}  \t| noisy level: {:.1f}".format(noise_level[0], noise_level[1], noise_level[2]))
print("                            \t|mean      \t  std  \t|mean      \t std\t|mean      \t std\t|mean      \t std")
print("-------------------------------------------------------------------------------------------------------------------------------")
print_line(values, feature_name="spectral flatness output    ", feature_index=sfo)
print_line(values, feature_name="spectral flatness difference", feature_index=sfd)
print_line(values, feature_name="average std changes k-means ", feature_index=stdmean)
print_line(values, feature_name="minimum std changes k-means ", feature_index=stdmin)
print_line(values, feature_name="maximum std changes k-means ", feature_index=stdmax)
print_line(values, feature_name="average drifts k-means      ", feature_index=drmean)
print_line(values, feature_name="minimum drifts k-means      ", feature_index=drmin)
print_line(values, feature_name="maximum drifts k-means      ", feature_index=drmax)
print("-------------------------------------------------------------------------------------------------------------------------------")