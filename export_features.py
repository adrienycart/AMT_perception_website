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
from features.polyphony import polyphony_level_seq, false_negative_polyphony_level
from features.repeat_merge import repeated_notes, merged_notes
from features.specific_pitch import specific_pitch_framewise, specific_pitch_notewise
from features.rhythm import rhythm_histogram

import cPickle as pickle


MIDI_path = 'app/static/data/all_midi_cut'
systems = ['kelz', 'lisu', 'google', 'cheng']

write_path = 'precomputed_features'

for example in os.listdir(MIDI_path):
    if example.startswith('.'):
        continue
    example_path = os.path.join(MIDI_path, example)  # folder path
    print('\n\npath = ' + example_path)
    target_data = pm.PrettyMIDI(os.path.join(example_path, 'target.mid'))
    dir = os.path.join(write_path,example)
    if not os.path.exists(dir):
        os.mkdir(dir)

    for system in systems:

        system_data = pm.PrettyMIDI(os.path.join(example_path, system + '.mid'))

        # target and system piano rolls
        # print('getting piano roll...')
        target_pr = (target_data.get_piano_roll()>0).astype(int)
        system_pr = (system_data.get_piano_roll()>0).astype(int)
        target_pr, system_pr = utils.even_up_rolls(target_pr, system_pr)

        notes_target, intervals_target, vel_target = utils.get_notes_intervals(target_data, with_vel=True)
        notes_system, intervals_system = utils.get_notes_intervals(system_data)

        frame = framewise(system_pr,target_pr)

        match_on = mir_eval.transcription.match_notes(intervals_target, notes_target, intervals_system, notes_system, offset_ratio=None, pitch_tolerance=0.25)
        match_onoff = mir_eval.transcription.match_notes(intervals_target, notes_target, intervals_system, notes_system, offset_ratio=0.2, pitch_tolerance=0.25)

        match_on_25 = mir_eval.transcription.match_notes(intervals_target, notes_target, intervals_system, notes_system, onset_tolerance = 0.025, offset_ratio=None, pitch_tolerance=0.25)
        match_onoff_25 = mir_eval.transcription.match_notes(intervals_target, notes_target, intervals_system, notes_system, onset_tolerance = 0.025, offset_ratio=0.2, pitch_tolerance=0.25)

        match_on_75 = mir_eval.transcription.match_notes(intervals_target, notes_target, intervals_system, notes_system, onset_tolerance = 0.075, offset_ratio=None, pitch_tolerance=0.25)
        match_onoff_75 = mir_eval.transcription.match_notes(intervals_target, notes_target, intervals_system, notes_system, onset_tolerance = 0.075, offset_ratio=0.2, pitch_tolerance=0.25)

        match_on_100 = mir_eval.transcription.match_notes(intervals_target, notes_target, intervals_system, notes_system, onset_tolerance = 0.01, offset_ratio=None, pitch_tolerance=0.25)
        match_onoff_100 = mir_eval.transcription.match_notes(intervals_target, notes_target, intervals_system, notes_system, onset_tolerance = 0.01, offset_ratio=0.2, pitch_tolerance=0.25)

        notewise_On_50 = notewise(match_on,notes_system,notes_target)
        notewise_OnOff_50 = notewise(match_onoff,notes_system,notes_target)

        notewise_On_25 = notewise(match_on_25,notes_system,notes_target)
        notewise_OnOff_25 = notewise(match_onoff_25,notes_system,notes_target)

        notewise_On_75 = notewise(match_on_75,notes_system,notes_target)
        notewise_OnOff_75 = notewise(match_onoff_75,notes_system,notes_target)

        notewise_On_100 = notewise(match_on_100,notes_system,notes_target)
        notewise_OnOff_100 = notewise(match_onoff_100,notes_system,notes_target)

        results_dict = {
                "framewise" : frame,
                "notewise_On_50": notewise_On_50,
                "notewise_OnOff_50":notewise_OnOff_50,
                "notewise_On_25": notewise_On_25,
                "notewise_OnOff_25": notewise_OnOff_25,
                "notewise_On_75": notewise_On_75,
                "notewise_OnOff_75":notewise_OnOff_75,
                "notewise_On_100":notewise_On_100,
                "notewise_OnOff_100":notewise_OnOff_100,
                }


        pickle.dump(results_dict, open(os.path.join(dir,system+'.pkl'), 'wb'))
