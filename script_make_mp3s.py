import pickle
import numpy as np
import pretty_midi as pm
import os
import copy
import shutil
import random
import matplotlib.pyplot as plt
import csv
import pydub
import wave
from utils import *


############################################################
#### Create separate cut files for each MIDI
############################################################


# with open('data/all_cut_points_pm.csv', 'rU') as csvfile:
#     readCSV = csv.reader(csvfile, delimiter=',', dialect=csv.excel_tab)
#     for i, row in enumerate(readCSV):
#         if not i==0 and not row[0]=='':
#             filename = row[0]
#             print filename
#             n_cuts = int(row[1])
#             matrix = np.zeros([n_cuts,2],dtype=object)
#             for i,string in enumerate(row[2:2+n_cuts]):
#                 start_end = string.split(';')
#                 matrix[i,0]=start_end[0]
#                 matrix[i,1]=start_end[1]
#             np.savetxt(os.path.join('data/cut_points',filename+'.csv'),matrix, fmt="%s")

############################################################
#### Prepare Onsets and Frames midi files
############################################################

# MAPS_folder = "data/MAPS_wav"
# dest_folder = "data/onsets_and_frames_outputs"
# for filename in os.listdir(MAPS_folder):
#     if filename.endswith('.midi') and not filename.startswith('.'):
#         maps_filename = os.path.join(MAPS_folder,filename)
#         dest_filename = os.path.join(dest_folder,filename.replace('.wav.midi','.mid'))
#         shutil.move(maps_filename,dest_filename)

# dest_folder = "data/onsets_and_frames_outputs"
# for filename in os.listdir(dest_folder):
#     if filename.endswith('.mid') and not filename.startswith('.'):
#         midi_path = os.path.join(dest_folder,filename)
#         data = pm.PrettyMIDI(midi_path)
#         for instr in data.instruments:
#             for note in instr.notes:
#                 note.velocity = 100
#         data.write(midi_path)



############################################################
#### Prepare Bittner midi files
############################################################

# csv_folder = 'data/bittner_csv'
# dest_folder = 'data/bittner_outputs'
# for filename in os.listdir(csv_folder):
#     if filename.endswith('.csv') and not filename.startswith('.'):
#         csv_filename = os.path.join(csv_folder,filename)
#         print filename
#         with open(csv_filename) as csvfile:
#             readCSV = csv.reader(csvfile, delimiter='\t')
#             piano_roll = []
#             times = []
#             for row in readCSV:
#                 times += [float(row[0])]
#                 frame = np.zeros([1,128])
#                 for freq in row[1:]:
#                     freq_quant = int(round(pm.hz_to_note_number(float(freq))))
#                     frame[0,freq_quant] = 1
#                 piano_roll += [frame]
#             piano_roll = np.concatenate(piano_roll).T
#             # print piano_roll.shape
#             # plt.imshow(piano_roll.T,aspect='auto',origin='lower')
#             # plt.show(block=[bool])
#
#             midi_data = make_midi_from_roll(piano_roll,None,times=times)
#             save_midi(midi_data,os.path.join(dest_folder,filename.replace("_multif0_multif0.csv",".mid")))

# ##############################################################
# #### Prepare Kelz MIDI files
# ##############################################################

# input_folder = '../../MLM_decoding/data/outputs_default_config_split/test'
# dest_folder = 'app/static/data/kelz_outputs'
#
# for filename in os.listdir(input_folder):
#     if filename.endswith('.csv') and not filename.startswith('.'):
#         csv_filename = os.path.join(input_folder,filename)
#
#         roll = np.transpose(np.loadtxt(csv_filename),[1,0])
#         roll_bin = (roll>0.5).astype(int)
#
#
#         roll_filt = filter_short_gaps(roll_bin,thresh=2)
#         roll_filt = filter_short_notes(roll_filt,thresh=2)
#
#         # fig,[ax1,ax2,ax3] = plt.subplots(3,1)
#         # ax1.imshow(roll_bin,origin='lower',aspect='auto')
#         # ax2.imshow(roll_filt,origin='lower',aspect='auto')
#         # ax3.imshow(roll_bin-roll_filt,origin='lower',aspect='auto',cmap=plt.get_cmap('seismic'))
#         # plt.show()
#
#         midi_data = make_midi_from_roll(roll_filt,fs=25,note_range=[21,109])
#         save_midi(midi_data,os.path.join(dest_folder,filename.replace(".csv",".mid")))

# ##############################################################
# #### Prepare Li Su or Cheng MIDI files
# ##############################################################

# input_folder = 'app/static/data/cheng_csv'
# dest_folder = 'app/static/data/cheng_outputs'
#
# for filename in os.listdir(input_folder):
#     if filename.endswith('.csv') and not filename.startswith('.'):
#         csv_filename = os.path.join(input_folder,filename)
#         all_notes = np.loadtxt(csv_filename)
#
#         midi_data = pm.PrettyMIDI()
#         piano_program = pm.instrument_name_to_program('Acoustic Grand Piano')
#         piano = pm.Instrument(program=piano_program)
#
#         for (start,end,freq) in all_notes:
#             if 'cheng' in input_folder:
#                 pitch = int(round(12*np.log2(freq/27.5)+1))
#                 # print(pitch,int(round(pm.hz_to_note_number(freq))))
#             elif "lisu" in input_folder:
#                 pitch = int(round(pm.hz_to_note_number(freq)))
#
#
#             if 0<=pitch < 128:
#                 note = pm.Note(
#                     velocity=100, pitch=pitch, start=start, end=end)
#                 piano.notes.append(note)
#             else:
#                 print "Removed pitch", pitch, "freq", freq
#         midi_data.instruments.append(piano)
#
#         dest_filename = os.path.join(dest_folder,filename).replace('.csv','.mid')
#         midi_data.write(dest_filename)


##############################################################
#### Apply pedal to MIDI files
##############################################################

# MAPS_folder = "app/static/data/MAPS_wav"
# AMAPS_folder = "app/static/data/A-MAPS_1.2"
# dest_folder = 'app/static/data/A-MAPS_1.2_with_pedal'
#
# for filename in os.listdir(MAPS_folder):
#     if filename.endswith('.wav') and not filename.startswith('.') and "chpn-e01" not in filename:# and 'MAPS_MUS-schu_143_1_ENSTDkAm' in filename:
#         filename = filename.replace('.wav','.mid')
#         print(filename)
#
#         amaps_filename = os.path.join(AMAPS_folder,filename)
#
#         midi = pm.PrettyMIDI(amaps_filename)
#         new_midi = apply_sustain_control_changes(midi)
#
#         new_midi.write(os.path.join(dest_folder,filename))

# ##############################################################
# #### Verify segments
# ##############################################################

# MAPS_folder = "app/static/data/MAPS_wav"
# AMAPS_folder = "app/static/data/A-MAPS_1.2_with_pedal"
# PM_folder = "../../MLM_decoding/data/piano-midi-ttv-20p/test"
# csv_folder = 'app/static/data/cut_points'
#
# fd = open("app/static/data/data_OK.txt", "r")
# OK_files = fd.read().splitlines()
# # print(OK_files)
# for filename in os.listdir(MAPS_folder):
#     if filename.endswith('.wav') and not filename.startswith('.') and "chpn-e01" not in filename and filename not in OK_files:# and 'mz_333_3_ENSTDkCl' in filename:
#         print(filename)
#         amaps_filename = os.path.join(AMAPS_folder,filename.replace('.wav','.mid'))
#
#         piece_name = get_name_from_maps(filename)
#
#         csv_filename = os.path.join(csv_folder,piece_name+'.csv')
#         pm_filename = os.path.join(PM_folder,piece_name+'.mid')
#
#
#         amaps_data = pm.PrettyMIDI(amaps_filename)
#         pm_data = pm.PrettyMIDI(pm_filename)
#
#         cut_points = np.genfromtxt(csv_filename,dtype='str')
#
#         for i,(start_str, end_str) in enumerate(cut_points):
#             start_bar,start_beat,start_sub_beat = str_to_bar_beat(start_str)
#             end_bar,end_beat,end_sub_beat = str_to_bar_beat(end_str)
#
#             pm_start_t = get_time(pm_data,start_bar,start_beat,start_sub_beat)
#             pm_end_t = get_time(pm_data,end_bar,end_beat,end_sub_beat)
#             pm_cut = cut_midi(pm_data,pm_start_t,pm_end_t)
#
#             amaps_start_t = get_time(amaps_data,start_bar,start_beat,start_sub_beat)
#             amaps_end_t = get_time(amaps_data,end_bar,end_beat,end_sub_beat)
#             amaps_cut = cut_midi(amaps_data,amaps_start_t,amaps_end_t)
#
#             pm_roll = (pm_cut.get_piano_roll()>0).astype(int)
#             amaps_roll = (amaps_cut.get_piano_roll()>0).astype(int)
#
#             pm_roll, amaps_roll = even_up_rolls(pm_roll, amaps_roll)
#
#             F_measure = np.sum(2*pm_roll*amaps_roll)/(np.sum(2*pm_roll*amaps_roll+np.abs(pm_roll-amaps_roll))+1e-7)
#
#             if F_measure < 0.80:
#                 print(filename,i,start_str,end_str ,F_measure)
#                 print(pm_start_t, pm_end_t)
#                 print(amaps_start_t, amaps_end_t)
#                 fig,[ax1,ax2,ax3] = plt.subplots(3,1)
#                 ax1.imshow(pm_roll,origin='lower',aspect='auto')
#                 ax2.imshow(amaps_roll,origin='lower',aspect='auto')
#                 for instr in pm_cut.instruments:
#                     for cc in instr.control_changes:
#                         if cc.number == 64:
#                             color = 'green' if cc.value > 64 else 'red'
#                             ax1.plot([int(round(cc.time*100)),int(round(cc.time*100))],[0,127],color=color)
#                 ax3.imshow(amaps_roll-pm_roll,origin='lower',aspect='auto',cmap=plt.get_cmap('seismic'))
#                 plt.show()
#         f=open("app/static/data/data_OK.txt", "a+")
#         f.write(filename+'\n')




##############################################################
#### Cut A-MAPS MIDI files into segments
##############################################################

# MAPS_folder = "app/static/data/MAPS_wav"
# AMAPS_folder = "app/static/data/A-MAPS_1.2_with_pedal"
# MIDI_input_folders = [
#                     # "app/static/data/cheng_outputs",
#                     # "app/static/data/onsets_and_frames_outputs",
#                     # "app/static/data/kelz_outputs",
#                     # "app/static/data/lisu_outputs"
#                     # "app/static/data/A-MAPS_1.2"
#                     ]
# MIDI_names = [
#                 # "cheng",
#                 # "google",
#                 # "kelz",
#                 # "lisu",
#                 # "target_no_pedal",
#                 ]
# csv_folder = 'app/static/data/cut_points'
# dest_folder = 'app/static/data/all_midi_cut'
# cut_points_seconds_folder = 'app/static/data/cut_points_seconds'
#
# write_AMAPS = False
#
# for filename in os.listdir(AMAPS_folder):
#     if filename.endswith('.mid') and not filename.startswith('.') and "chpn-e01" not in filename:# and 'MAPS_MUS-liz_rhap02_ENSTDkAm' in filename:
#
#         filename = filename.replace('.wav','.mid')
#         print(filename)
#
#         piece_name = get_name_from_maps(filename)
#         csv_filename = os.path.join(csv_folder,piece_name+'.csv')
#         amaps_filename = os.path.join(AMAPS_folder,filename)
#
#         amaps_data = pm.PrettyMIDI(amaps_filename)
#         cut_points = np.genfromtxt(csv_filename,dtype='str')
#
#         midis = []
#         for input_folder in MIDI_input_folders:
#              midi = pm.PrettyMIDI(os.path.join(input_folder,filename))
#              midis += [midi]
#
#         cut_points_seconds = []
#
#         for i, (start_str, end_str) in enumerate(cut_points):
#
#             save_folder = os.path.join(dest_folder,os.path.splitext(filename)[0]+'_'+str(i))
#             safe_mkdir(save_folder)
#
#             start_bar,start_beat,start_sub_beat = str_to_bar_beat(start_str)
#             end_bar,end_beat,end_sub_beat = str_to_bar_beat(end_str)
#
#             start_t = get_time(amaps_data,start_bar,start_beat,start_sub_beat)
#             end_t = get_time(amaps_data,end_bar,end_beat,end_sub_beat)
#
#             assert start_t < end_t
#
#             cut_points_seconds += [[start_t,end_t]]
#
#             if write_AMAPS:
#                 amaps_data_cut = cut_midi(amaps_data,start_t,end_t)
#                 amaps_data_cut.write(os.path.join(save_folder,'target.mid'))
#
#             for midi,name in zip(midis,MIDI_names):
#                 midi_cut = cut_midi(midi,start_t,end_t)
#                 midi_cut.write(os.path.join(save_folder,name+'.mid'))
#
#             # f= open(os.path.join(save_folder,"duration.txt"),"w+")
#             # f.write(str(end_t-start_t))
#             # f.close()
#
#         cut_points_seconds = np.array(cut_points_seconds)
#         np.savetxt(os.path.join(cut_points_seconds_folder,filename.replace('.mid','.csv')),cut_points_seconds)
#


# ##############################################################
# #### Convert MIDI files into mp3 files
# ##############################################################
#
# midi_folder = 'app/static/data/all_midi_cut'
# dest_folder = 'app/static/data/all_mp3_cut'
# csv_folder = 'app/static/data/cut_points'
# AMAPS_folder = "app/static/data/A-MAPS_1.2"
#
# to_recompute = ['kelz']
#
# all_midi_folders = os.listdir(midi_folder)
# n_midi_folders = len(all_midi_folders)
# for i,subfolder_name in enumerate(all_midi_folders):
#     subfolder = os.path.join(midi_folder,subfolder_name)
#
#     if os.path.isdir(subfolder):# and 'MAPS_MUS-mz_331_2_ENSTDkCl_16' in subfolder:
#         print('(',i,'/',n_midi_folders,')', subfolder)
#         dest_subfolder = os.path.join(dest_folder,subfolder_name)
#         safe_mkdir(dest_subfolder)
#
#         # #Retrieve duration of example
#         f=open(os.path.join(subfolder,"duration.txt"), "r")
#         dur_str = f.read()
#         duration = float(dur_str)
#
#         for midi_file in os.listdir(subfolder):
#             if midi_file.endswith('.mid') and not midi_file.startswith('.'):
#                 wav_path = os.path.join(dest_subfolder,midi_file.replace('.mid','.wav'))
#                 if not os.path.exists(wav_path.replace('.wav','.mp3')) or os.path.splitext(midi_file)[0] in to_recompute:
#
#                     data = pm.PrettyMIDI(os.path.join(subfolder,midi_file))
#                     # print midi_file
#                     # for instr in data.instruments:
#                     #     print instr.control_changes
#
#                     sound1 = synthesize_midi(data)
#                     sound1_trim = sound1[:int(duration*44100)]
#                     wav_path = os.path.join(dest_subfolder,midi_file.replace('.mid','.wav'))
#                     write_sound(sound1_trim,wav_path)
#
#                     sound2 = pydub.AudioSegment.from_wav(wav_path)
#                     sound2.export(wav_path.replace('.wav','.mp3'), format="mp3", bitrate="320k")
#                     os.remove(wav_path)
