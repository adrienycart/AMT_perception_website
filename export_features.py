import os
import numpy as np
import pretty_midi as pm
import mir_eval
import matplotlib.pyplot as plt
import features.utils as utils
from features.benchmark import framewise, notewise
from features.high_low_voice import framewise_highest, framewise_lowest, notewise_highest, notewise_lowest
from features.loudness import false_negative_loudness, loudness_ratio_false_negative
from features.out_key import make_key_mask, out_key_errors, out_key_errors_binary_mask
from features.polyphony import polyphony_level_diff
from features.repeat_merge import repeated_notes, merged_notes
from features.specific_pitch import specific_pitch_framewise, specific_pitch_notewise
from features.rhythm import rhythm_histogram, rhythm_dispersion

import warnings
warnings.filterwarnings("ignore")

try:
    # python 2.7
    import cPickle as pickle
except:
    # python 3
    import pickle

MIDI_path = 'app/static/data/all_midi_cut'
systems = ['kelz', 'lisu', 'google', 'cheng']

fs=100

precomputed_consonance_path = 'features/consonance_statistics'

write_path = 'precomputed_features'
example_paths = [path for path in os.listdir(MIDI_path) if not path.startswith('.')]
for i,example in enumerate(example_paths):
    if example.startswith('.'):# or not 'MAPS_MUS-mendel_op62_5_ENSTDkAm_13' in example:
        continue
    example_path = os.path.join(MIDI_path, example)  # folder path
    print(str(i)+'/'+str(len(example_paths))+' path = ' + example_path)
    target_data = pm.PrettyMIDI(os.path.join(example_path, 'target.mid'))
    target_data_no_pedal = pm.PrettyMIDI(os.path.join(example_path, 'target_no_pedal.mid'))
    dir = os.path.join(write_path,example)
    if not os.path.exists(dir):
        os.mkdir(dir)

    for system in systems:
        results_dict = {}

        system_data = pm.PrettyMIDI(os.path.join(example_path, system + '.mid'))

        # target and system piano rolls
        # print('getting piano roll...')
        target = (target_data.get_piano_roll(fs)>0).astype(int)
        try:
            # if pretty_midi is installed using the latest sources, it takes into account pedal by default
            target_no_pedal = (target_data_no_pedal.get_piano_roll(fs,pedal_threshold=None)>0).astype(int)
        except:
            # if pretty_midi is installed using pip, it does not, pedal_threshold parameter doesn't exist.
            target_no_pedal = (target_data_no_pedal.get_piano_roll(fs)>0).astype(int)
        output = (system_data.get_piano_roll(fs)>0).astype(int)
        target,target_no_pedal, output = utils.even_up_rolls([target, target_no_pedal, output])

        notes_target, intervals_target, vel_target = utils.get_notes_intervals(target_data, with_vel=True)
        notes_target_no_pedal, intervals_target_no_pedal, vel_target_no_pedal = utils.get_notes_intervals(target_data_no_pedal, with_vel=True)
        notes_output, intervals_output = utils.get_notes_intervals(system_data)

        frame = framewise(output,target)

        #### Investigate various frame sizes
        for f in [0.05,0.075,0.1,0.15]:
            times = np.arange(0,max(target_data.get_end_time(),system_data.get_end_time()),f)
            roll_target = utils.get_roll_from_times(target_data,times)
            roll_output = utils.get_roll_from_times(system_data,times)
            result = framewise(roll_target,roll_output)
            results_dict.update({'framewise_'+str(f): result})

        for on_tol in [25,50,75,100,125,150]:
            match_on = mir_eval.transcription.match_notes(intervals_target, notes_target, intervals_output, notes_output, onset_tolerance=on_tol/1000.0, offset_ratio=None, pitch_tolerance=0.25)
            if on_tol == 50:
                match = match_on
            note = notewise(match_on,notes_output,notes_target)
            results_dict.update({'notewise_On_'+str(on_tol): note})

        for on_tol in [25,50,75,100,125,150]:
            for off_tol in [0.1,0.2,0.3,0.4,0.5]:
                match_onoff = mir_eval.transcription.match_notes(intervals_target, notes_target, intervals_output, notes_output,onset_tolerance=on_tol/1000.0, offset_ratio=off_tol, pitch_tolerance=0.25)
                note = notewise(match_onoff,notes_output,notes_target)
                results_dict.update({'notewise_OnOff_'+str(on_tol)+'_'+str(object=off_tol): note})


        high_f = framewise_highest(output, target_no_pedal)
        low_f = framewise_lowest(output, target_no_pedal)

        high_n = notewise_highest(notes_output, intervals_output, notes_target_no_pedal, intervals_target_no_pedal, match)
        low_n = notewise_lowest(notes_output, intervals_output, notes_target_no_pedal, intervals_target_no_pedal, match)

        loud_fn = false_negative_loudness(match, vel_target, intervals_target)
        loud_ratio_fn = loudness_ratio_false_negative(notes_target, intervals_target, vel_target, match)

        mask = make_key_mask(target_no_pedal)
        out_key = out_key_errors(notes_output, match, mask)
        out_key_bin = out_key_errors_binary_mask(notes_output, match, mask)

        repeat = repeated_notes(notes_output, intervals_output, notes_target, intervals_target, match)
        merge = merged_notes(notes_output, intervals_output, notes_target, intervals_target, match)

        semitone_f = specific_pitch_framewise(output, target, fs, 1)
        octave_f = specific_pitch_framewise(output, target, fs, 12)
        third_harmonic_f = specific_pitch_framewise(output, target, fs, 19,down_only=True)

        semitone_n = specific_pitch_notewise(notes_output, intervals_output, notes_target, intervals_target, match_on, n_semitones=1)
        octave_n = specific_pitch_notewise(notes_output, intervals_output, notes_target, intervals_target, match_on, n_semitones=12)
        third_harmonic_n = specific_pitch_notewise(notes_output, intervals_output, notes_target, intervals_target, match_on, n_semitones=19,down_only=True)

        poly_diff = polyphony_level_diff(output,target)

        rhythm_hist = rhythm_histogram(intervals_output,intervals_target)
        rhythm_disp_std,rhythm_disp_drift = rhythm_dispersion(intervals_output, intervals_target)

        #Import consonance features
        consonance_target = pickle.load(open(os.path.join(precomputed_consonance_path,example,'target.pkl'),'rb'))
        consonance_output = pickle.load(open(os.path.join(precomputed_consonance_path,example,system+'.pkl'),'rb'))

        cons_hut78_target = [consonance_target[0],consonance_target[3],consonance_target[6],consonance_target[9]]
        cons_har18_target = [consonance_target[1],consonance_target[4],consonance_target[7],consonance_target[10]]
        cons_har19_target = [consonance_target[2],consonance_target[5],consonance_target[8],consonance_target[11]]

        cons_hut78_output = [consonance_output[0],consonance_output[3],consonance_output[6],consonance_output[9]]
        cons_har18_output = [consonance_output[1],consonance_output[4],consonance_output[7],consonance_output[10]]
        cons_har19_output = [consonance_output[2],consonance_output[5],consonance_output[8],consonance_output[11]]

        cons_hut78_diff = [c1-c2 for (c1,c2) in zip(cons_hut78_output,cons_hut78_target)]
        cons_har18_diff = [c1-c2 for (c1,c2) in zip(cons_har18_output,cons_har18_target)]
        cons_har19_diff = [c1-c2 for (c1,c2) in zip(cons_har19_output,cons_har19_target)]

        results_dict.update({
                "framewise_0.01" : frame,

                #Notewise have already been added

                "high_f": high_f,
                "low_f": low_f,
                "high_n": high_n,
                "low_n":low_n,

                "loud_fn":loud_fn,
                "loud_ratio_fn":loud_ratio_fn,

                "out_key":out_key,
                "out_key_bin":out_key_bin,

                "repeat":repeat,
                "merge":merge,

                "semitone_f":semitone_f,
                "octave_f":octave_f,
                "third_harmonic_f":third_harmonic_f,
                "semitone_n":semitone_n,
                "octave_n":octave_n,
                "third_harmonic_n":third_harmonic_n,

                "poly_diff": poly_diff,

                "rhythm_hist": rhythm_hist,
                "rhythm_disp_std": rhythm_disp_std,
                "rhythm_disp_drift": rhythm_disp_drift,

                "cons_hut78_output": cons_hut78_output,
                "cons_har18_output": cons_har18_output,
                "cons_har19_output":  cons_har19_output,

                "cons_hut78_diff" : cons_hut78_diff,
                "cons_har18_diff" : cons_har18_diff,
                "cons_har19_diff" : cons_har19_diff,

                })


        pickle.dump(results_dict, open(os.path.join(dir,system+'.pkl'), 'wb'),protocol=2)
