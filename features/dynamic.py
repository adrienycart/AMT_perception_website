import numpy as np
import os
from statistics import stdev
from .utils import create_folder
from dissonant import harmonic_tone, dissonance, pitch_to_freq

def get_pitch(full_note):
    return full_note[0]

def get_onset(full_note):
    return full_note[1]

def get_offset(full_note):
    return full_note[2]

def weighted_std(values, weighted_mean, weights):
    std = np.sqrt(sum([(values[idx] - weighted_mean)**2 * weights[idx] for idx in range(len(values))]) / sum(weights))
    return std

def pad_chords(chords):
    max_poly = max(len(chord) for chord in chords)
    for chord in chords:
        chord.extend([-1] * (max_poly - len(chord)))
    print(np.array(chords))
    return chords

def unpad_chords(chords):
    unpad = []
    for chord in chords:
        unpad.append([p for p in chord if p != -1])
    return unpad

def get_event_based_sequence(notes, intervals, example, system, dt=0.05):

    folder = "features/chords_and_times/" + example + "/"

    # if precalculated, simple load values
    if os.path.isfile(folder + system + "_chords.npy"):
        chords = np.load(folder + system + "_chords.npy", allow_pickle=True)
        chords = [list(chords[i]) for i in range(chords.shape[0])]
        chords = unpad_chords(chords)
        event_times = list(np.load(folder + system + "_event_times.npy", allow_pickle=True))
        durations = list(np.load(folder + system + "_durations.npy", allow_pickle=True))
        return chords, event_times, durations

    # if not pre-calculated, calculate chords and event times.
    full_notes = [(notes[idx], intervals[idx][0], intervals[idx][1]) for idx in range(len(notes))]
    full_notes.sort(key=get_onset)
    all_times = []
    for interval in intervals:
        all_times.extend(interval)
    all_times.sort()

    # get event times from the onsets and offsets
    last_time = -1.0
    event_times = []
    for time in all_times:
        if time - last_time > dt:
            event_times.append(time)
            last_time = time

    chords = []
    for idx in range(len(event_times)-1):
        chord = []
        for note in full_notes:
            if get_onset(note) < event_times[idx] + dt and get_offset(note) > event_times[idx+1] - dt:
                chord.append(get_pitch(note))
        chords.append(chord)

    # from utils import make_note_index_matrix
    # matrix = make_note_index_matrix(notes, intervals)
    # import matplotlib.pyplot as plt
    # plt.imshow(matrix)
    # plt.show()

    durations = [event_times[idx+1] - event_times[idx] for idx in range(len(chords))]

    create_folder(folder)
    np.save(folder + system + "_chords.npy", np.array(pad_chords(chords)))
    np.save(folder + system + "_event_times.npy", np.array(event_times))
    np.save(folder + system + "_durations.npy", np.array(durations))
    # print(system + 'saved.')

    return chords, event_times, durations


def chord_dissonance(notes_output, intervals_output, notes_target, intervals_target, example, system):

    chords_target, event_times_target, durations_target = get_event_based_sequence(notes_target, intervals_target, example, "target")
    chords_output, event_times_output, durations_output = get_event_based_sequence(notes_output, intervals_output, example, system)

    dissonances_target = []
    for chord in chords_target:
        if len(chord) == 0:
            dissonances_target.append(0.0)
        else:
            freqs, amps = harmonic_tone(pitch_to_freq(chord), n_partials=10)
            dissonances_target.append(dissonance(freqs, amps, model='sethares1993'))

    dissonances_output = []
    for chord in chords_output:
        if len(chord) == 0:
            dissonances_output.append(0.0)
        else:
            freqs, amps = harmonic_tone(pitch_to_freq(chord), n_partials=10)
            dissonances_output.append(dissonance(freqs, amps, model='sethares1993'))

    ave_dissonance_target = np.average(dissonances_target, weights=durations_target)
    ave_dissonance_output = np.average(dissonances_output, weights=durations_output)

    std_dissonance_target = weighted_std(dissonances_target, ave_dissonance_target, durations_target)
    std_dissonance_output = weighted_std(dissonances_output, ave_dissonance_output, durations_output)

    return (
        ave_dissonance_target,
        ave_dissonance_output,
        std_dissonance_target,
        std_dissonance_output,
        max(dissonances_target),
        max(dissonances_output),
        min(x for x in dissonances_target if x > 0.0),
        min(x for x in dissonances_output if x > 0.0)
    )


def polyphony_level(notes_output, intervals_output, notes_target, intervals_target, example, system):

    chords_target, event_times_target, durations_target = get_event_based_sequence(notes_target, intervals_target, example, "target")
    chords_output, event_times_output, durations_output = get_event_based_sequence(notes_output, intervals_output, example, system)
    print(chords_target)

    polyphony_levels_target = [len(chord) for chord in chords_target]
    polyphony_levels_output = [len(chord) for chord in chords_output]

    # weighted averages and stds
    ave_polyphony_level_target = np.average(polyphony_levels_target, weights=durations_target)
    ave_polyphony_level_output = np.average(polyphony_levels_output, weights=durations_output)
    std_polyphony_level_target = weighted_std(polyphony_levels_target, ave_polyphony_level_target, durations_target)
    std_polyphony_level_output = weighted_std(polyphony_levels_output, ave_polyphony_level_output, durations_output)

    # unweighted mean and stds.
    mean_target = np.mean(polyphony_levels_target)
    mean_output = np.mean(polyphony_levels_output)
    std_target = stdev(polyphony_levels_target)
    std_output = stdev(polyphony_levels_output)

    # import matplotlib.pyplot as plt
    # plt.plot(event_times_target, polyphony_levels_target + [0], label='target')
    # plt.plot(event_times_output, polyphony_levels_output + [0], label='output')
    # plt.legend()
    # plt.show()

    return (
        # weighted
        ave_polyphony_level_target,
        ave_polyphony_level_output,
        std_polyphony_level_target,
        std_polyphony_level_output,
        # unweighted
        mean_target,
        mean_output,
        std_target,
        std_output,
        max(polyphony_levels_target),
        max(polyphony_levels_output),
        min(polyphony_levels_target),
        min(polyphony_levels_output)
    )
