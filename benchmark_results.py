import os
import numpy as np
import pretty_midi as pm
import mir_eval
import matplotlib.pyplot as plt
import features.utils as utils
from features.benchmark import framewise, notewise


data_folder = 'app/static/data'
subfolders= ['cheng_outputs','kelz_outputs','lisu_outputs','onsets_and_frames_outputs']
GT_folder = 'app/static/data/A-MAPS_1.2_with_pedal'



fs=100

for fold in subfolders:
    results_frame_Cl = []
    results_note_on_Cl = []
    results_note_onoff_Cl = []
    results_frame_Am = []
    results_note_on_Am = []
    results_note_onoff_Am = []
    subfolder_path = os.path.join(data_folder,fold)

    example_paths = [path for path in os.listdir(subfolder_path) if not path.startswith('.') and path.endswith('.mid') and not "MAPS_MUS-chpn-e01_ENSTDkCl" in path]
    for i,example in enumerate(example_paths):
        example_path = os.path.join(subfolder_path,example)
        target_path = os.path.join(GT_folder,example)
        print(str(i)+'/'+str(len(example_paths))+' path = ' + example_path)

        target_data = pm.PrettyMIDI(target_path)
        system_data = pm.PrettyMIDI(example_path)

        target = (target_data.get_piano_roll(fs)>0).astype(int)
        output = (system_data.get_piano_roll(fs)>0).astype(int)
        target, output = utils.even_up_rolls([target, output])

        notes_target, intervals_target, vel_target = utils.get_notes_intervals(target_data, with_vel=True)
        notes_output, intervals_output = utils.get_notes_intervals(system_data)

        frame = framewise(output,target)


        match_on = mir_eval.transcription.match_notes(intervals_target, notes_target, intervals_output, notes_output, onset_tolerance=0.05, offset_ratio=None, pitch_tolerance=0.25)
        note_on = notewise(match_on,notes_output,notes_target)

        match_onoff = mir_eval.transcription.match_notes(intervals_target, notes_target, intervals_output, notes_output,onset_tolerance=0.05, offset_ratio=0.2, pitch_tolerance=0.25)
        note_onoff = notewise(match_onoff,notes_output,notes_target)

        if "ENSTDkCl" in example:
            results_frame_Cl+= [frame]
            results_note_on_Cl+= [note_on]
            results_note_onoff_Cl+= [note_onoff]
        elif "ENSTDkAm" in example:
            results_frame_Am+= [frame]
            results_note_on_Am+= [note_on]
            results_note_onoff_Am+= [note_onoff]
        else:
            raise Exception('wut?')

    print subfolder_path
    print "ENSTDkCl:"
    print "frame:", np.round(np.mean(np.array(results_frame_Cl),axis=0)*100,1)
    print "note_on:", np.round(np.mean(np.array(results_note_on_Cl),axis=0)*100,1)
    print "note_onoff:", np.round(np.mean(np.array(results_note_onoff_Cl),axis=0)*100,1)
    print "ENSTDkAm:"
    print "frame:", np.round(np.mean(np.array(results_frame_Am),axis=0)*100,1)
    print "note_on:", np.round(np.mean(np.array(results_note_on_Am),axis=0)*100,1)
    print "note_onoff:", np.round(np.mean(np.array(results_note_onoff_Am),axis=0)*100,1)
    print "ALL:"
    print "frame:", np.round(np.mean(np.array(results_frame_Cl+results_frame_Am),axis=0)*100,1)
    print "note_on:", np.round(np.mean(np.array(results_note_on_Cl+results_note_on_Am),axis=0)*100,1)
    print "note_onoff:", np.round(np.mean(np.array(results_note_onoff_Cl+results_note_onoff_Am),axis=0)*100,1)


# #### CHENG
# ENSTDkCl:
# frame: [78.  71.9 74.2]
# note_on: [88.5 67.6 76.1]
# note_onoff: [43.7 34.  38. ]
# ENSTDkAm:
# frame: [64.9 55.1 58.8]
# note_on: [71.1 46.8 55.6]
# note_onoff: [28.  19.  22.3]
# ALL:
# frame: [71.3 63.3 66.4]
# note_on: [79.6 57.  65.7]
# note_onoff: [35.7 26.4 30. ]
#
# #### KELZ
# ENSTDkCl:
# frame: [88.9 55.  67.3]
# note_on: [76.3 57.1 64.2]
# note_onoff: [35.7 26.8 30.1]
# ENSTDkAm:
# frame: [89.3 51.4 64.3]
# note_on: [72.5 48.5 57.2]
# note_onoff: [27.1 19.  22. ]
# ALL:
# frame: [89.1 53.2 65.7]
# note_on: [74.4 52.8 60.6]
# note_onoff: [31.3 22.8 26. ]
#
# #### LISU
# ENSTDkCl:
# frame: [65.4 63.  63.6]
# note_on: [50.1 35.7 41.1]
# note_onoff: [18.2 13.5 15.3]
# ENSTDkAm:
# frame: [68.8 57.2 61.8]
# note_on: [49.5 28.5 35.6]
# note_onoff: [14.8  9.1 11.1]
# ALL:
# frame: [67.2 60.  62.7]
# note_on: [49.8 32.  38.3]
# note_onoff: [16.5 11.3 13.2]
#
# #### GOOGLE
# ENSTDkCl:
# frame: [87.8 79.2 83.1]
# note_on: [86.2 83.8 84.9]
# note_onoff: [67.8 65.9 66.8]
# ENSTDkAm:
# frame: [90.1 79.8 84.5]
# note_on: [85.5 84.4 84.9]
# note_onoff: [66.1 65.2 65.6]
# ALL:
# frame: [89.  79.5 83.8]
# note_on: [85.9 84.1 84.9]
# note_onoff: [66.9 65.5 66.2]
