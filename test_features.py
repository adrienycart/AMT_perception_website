import os
import time
import numpy as np
import pretty_midi as pm
import mir_eval
import matplotlib.pyplot as plt
import features.utils as utils
from features.high_low_voice import framewise_highest, framewise_lowest, notewise_highest, notewise_lowest, correct_highest_lowest_note_framewise
from features.loudness import false_negative_loudness, loudness_ratio_false_negative
from features.out_key import make_key_mask, out_key_errors, out_key_errors_binary_mask
from features.repeat_merge import repeated_notes, merged_notes
from features.specific_pitch import specific_pitch_framewise, specific_pitch_notewise
from features.rhythm import rhythm_histogram, rhythm_dispersion
from features.dynamic import chord_dissonance, polyphony_level
import warnings
warnings.filterwarnings("ignore")


MIDI_path = 'app/static/data/all_midi_cut'
systems = ['kelz', 'lisu', 'google', 'cheng']
fs = 100


for example in os.listdir(MIDI_path)[:12]:
    example_path = os.path.join(MIDI_path, example)  # folder path
    print('\n\npath = ' + example_path)

    target_data = pm.PrettyMIDI(os.path.join(example_path, 'target.mid'))
    target_pr = (target_data.get_piano_roll(fs)>0).astype(int)
    notes_target, intervals_target, vel_target = utils.get_notes_intervals(target_data, with_vel=True)

    target_data_no_pedal = pm.PrettyMIDI(os.path.join(example_path, 'target_no_pedal.mid'))
    target_pr_no_pedal = (target_data_no_pedal.get_piano_roll(fs)>0).astype(int)
    notes_target_no_pedal, intervals_target_no_pedal, vel_target_no_pedal = utils.get_notes_intervals(target_data_no_pedal, with_vel=True)

    # play midi
    # os.system("app\\static\\data\\all_midi_cut\\"+example+"\\target.mid")
    # time.sleep(target_data.get_end_time() + 0.5)

    for system in systems:
        if system == 'cheng':
            print('\n' + system)
            system_data = pm.PrettyMIDI(os.path.join(example_path, system + '.mid'))
            # play midi
            # os.system("app\\static\\data\\all_midi_cut\\"+example+"\\"+system+".mid")
            # time.sleep(system_data.get_end_time() + 0.5)

            system_pr = (system_data.get_piano_roll(fs)>0).astype(int)
            notes_system, intervals_system = utils.get_notes_intervals(system_data)

            target_pr, system_pr = utils.even_up_rolls(target_pr, system_pr)
            target_pr_no_pedal, system_pr_no_pedal = utils.even_up_rolls(target_pr_no_pedal, system_pr)

            if len(notes_system) == 0:
                match_on = []
                match_onoff = []
                match_on_no_pedal = []
                match_onoff_no_pedal = []
            else:
                # calculate true positives
                match_on = mir_eval.transcription.match_notes(intervals_target, notes_target, intervals_system, notes_system, offset_ratio=None, pitch_tolerance=0.25)
                match_onoff = mir_eval.transcription.match_notes(intervals_target, notes_target, intervals_system, notes_system, offset_ratio=0.2, pitch_tolerance=0.25)

                match_on_no_pedal = mir_eval.transcription.match_notes(intervals_target_no_pedal, notes_target_no_pedal, intervals_system, notes_system, offset_ratio=None, pitch_tolerance=0.25)
                match_onoff_no_pedal = mir_eval.transcription.match_notes(intervals_target_no_pedal, notes_target_no_pedal, intervals_system, notes_system, offset_ratio=0.2, pitch_tolerance=0.25)

            # test features...

            # print('test high_low_voice =================================================')

            # # high low voice framewise
            # highest_p, highest_r, highest_f = framewise_highest(system_pr_no_pedal, target_pr_no_pedal)
            # print('highest evaluation: ' + str(highest_p) + ', ' + str(highest_r) + ', ' + str(highest_f))
            # lowest_p, lowest_r, lowest_f = framewise_lowest(system_pr_no_pedal, target_pr_no_pedal)
            # print('lowest evaluation: ' + str(lowest_p) + ', ' + str(lowest_r) + ', ' + str(lowest_f))

            # # high low voice notewise
            # highest_p, highest_r, highest_f = notewise_highest(notes_system, intervals_system, notes_target_no_pedal, intervals_target_no_pedal, match_on_no_pedal)
            # print('highest evaluation: ' + str(highest_p) + ', ' + str(highest_r) + ', ' + str(highest_f))
            # lowest_p, lowest_r, lowest_f = notewise_lowest(notes_system, intervals_system, notes_target_no_pedal, intervals_target_no_pedal, match_on_no_pedal)
            # print('lowest evaluation: ' + str(lowest_p) + ', ' + str(lowest_r) + ', ' + str(lowest_f))

            # print('\n test loudness ========================================================')
            # value = false_negative_loudness(match_on, vel_target, intervals_target)
            # print('loudness value: ' + str(value))
            # ratio = loudness_ratio_false_negative(notes_target, intervals_target, vel_target, match_on)
            # print('loudness ratio: ' + str(ratio))

            # print('\n test out_key =======================================================')
            # mask = make_key_mask(target_pr)

            # print('\n>>>> non-binary pitch profile, onset only')
            # ratio_1, ratio_2 = out_key_errors(notes_system, match_on, mask)
            # print('ratios: ' + str(ratio_1) + ', ' + str(ratio_2))

            # print('\n>>>> binary pitch profile, onset only')
            # ratio_1, ratio_2 = out_key_errors_binary_mask(notes_system, match_on, mask)
            # print('ratios: ' + str(ratio_1) + ', ' + str(ratio_2))

            # print('\n test repeat_merge ===========================================')
            # repeat_ratio = repeated_notes(notes_system, intervals_system, notes_target, intervals_target, match_on)
            # print('repeated notes ratios: ' + str(repeat_ratio[0]) + ', ' + str(repeat_ratio[1]))
            # merge_ratio = merged_notes(notes_system, intervals_system, notes_target, intervals_target, match_on)
            # print('merged notes ratios: ' + str(merge_ratio[0]) + ', ' + str(merge_ratio[1]))

            # print('\n test rhythm =====================================================')
            # f1, f2 = rhythm_histogram(intervals_system, intervals_target)
            # print("logged spectral flatness: " + str(f1) + "(output)   " + str(f2) + "(target)")
            # print("rhythm flatness difference: " + str(f1-f2))
            # stds_change, drifts = rhythm_dispersion(intervals_system, intervals_target)
            # print("std changes: " + str(stds_change) + "\ndrifts: " + str(drifts))

            # print('\n test specific_pitch ==================================================')

            # print('\n>>framewise>>')
            # semitone = specific_pitch_framewise(system_pr, target_pr, fs, 1)
            # print('semitone error: ' + str(semitone))
            # octave = specific_pitch_framewise(system_pr, target_pr, fs, 12)
            # print('octave error: ' + str(octave))
            # third_harmonic = specific_pitch_framewise(system_pr, target_pr, fs, 19)
            # print('third_harmonic: ' + str(third_harmonic))

            # print('\n>>notewise>>')
            # r1, r2 = specific_pitch_notewise(notes_system, intervals_system, notes_target, intervals_target, match_on, n_semitones=1)
            # print('semitone error: ' + str(r1) + "   " + str(r2))
            # r1, r2 = specific_pitch_notewise(notes_system, intervals_system, notes_target, intervals_target, match_on, n_semitones=8)
            # print('octave error: ' + str(r1) + "   " + str(r2))
            # r1, r2 = specific_pitch_notewise(notes_system, intervals_system, notes_target, intervals_target, match_on, n_semitones=19)
            # print('third_harmonic error: ' + str(r1) + "   " + str(r2))

            print('\n test dynamic features ==================================================')
            # print('>> consonance measures')
            # print(consonance_measures(notes_system, intervals_system, notes_target, intervals_target, example, system))

            print('>> chord dissonance in the form of')
            print('(mean_target, mean_output, max_target, max_output, min_target, min_output):')
            print(chord_dissonance(notes_system, intervals_system, notes_target, intervals_target))

            # print('>> polyphony level in the form of')
            # print('(mean_target, mean_output, max_target, max_output, min_target, min_output):')
            # print(polyphony_level(notes_system, intervals_system, notes_target, intervals_target))
