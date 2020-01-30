import numpy as np

def get_pitch(full_note):
    return full_note[0]

def get_onset(full_note):
    return full_note[1]

def get_offset(full_note):
    return full_note[2]

def get_event_based_sequence(notes, intervals, dt=0.02):
    full_notes = [(notes[idx], intervals[idx][0], intervals[idx][1]) for idx in range(len(notes))]
    full_notes.sort(key=get_onset)

    last_onset = -1.0
    event_onsets = []
    for full_note in full_notes:
        if get_onset(full_note) - last_onset > dt:
            event_onsets.append(get_onset(full_note))
            last_onset = get_onset(full_note)
        # print(full_note)
    # print(event_onsets)

    chords = []
    for onset in event_onsets:
        chord = []
        for note in full_notes:
            if get_onset(note) < onset + dt and get_offset(note) > onset - dt:
                chord.append(get_pitch(note))
        chords.append(chord)
    # print(chords)

    event_times = event_onsets + [max([intervals[i][1] for i in range(len(intervals))])]
    # print(event_times)

    # from utils import make_note_index_matrix
    # matrix = make_note_index_matrix(notes, intervals)
    # import matplotlib.pyplot as plt
    # plt.imshow(matrix)
    # plt.show()

    return chords, event_times

def consonance_measures(notes_output, intervals_output, notes_target, intervals_target):

    chords_target, event_times_target = get_event_based_sequence(notes_target, intervals_target)


    return 0.0

def polyphony_level(notes_output, intervals_output, notes_target, intervals_target):

    chords_target, event_times_target = get_event_based_sequence(notes_target, intervals_target)
    chords_output, event_times_output = get_event_based_sequence(notes_output, intervals_output)

    polyphony_levels_target = [len(chord) for chord in chords_target]
    polyphony_levels_output = [len(chord) for chord in chords_output]
    # print(ave_polyphony_level_target)
    # print(ave_polyphony_level_output)

    ave_polyphony_level_target = np.mean([polyphony_levels_target[idx] * (event_times_target[idx+1] - event_times_target[idx]) for idx in range(len(chords_target))])
    ave_polyphony_level_output = np.mean([polyphony_levels_output[idx] * (event_times_output[idx+1] - event_times_output[idx]) for idx in range(len(chords_output))])

    # import matplotlib.pyplot as plt
    # plt.plot(event_times_target, polyphony_levels_target + [0], label='target')
    # plt.plot(event_times_output, polyphony_levels_output + [0], label='output')
    # plt.legend()
    # plt.show()

    return ave_polyphony_level_target, ave_polyphony_level_output, max(polyphony_levels_target), max(polyphony_levels_output), min(polyphony_levels_target), min(polyphony_levels_output)
