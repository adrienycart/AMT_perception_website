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
        target, output = utils.even_up_rolls(target, output)

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
