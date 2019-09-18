
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
import pyloudnorm as pyln

def safe_mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def even_up_rolls(roll1,roll2):
    #Makes roll1 and roll2 of same size.
    len_roll1 = roll1.shape[1]
    len_roll2 = roll2.shape[1]
    if len_roll1 > len_roll2:
        roll2 = np.concatenate([roll2,np.zeros([roll2.shape[0],len_roll1-len_roll2])],axis=1)
        # roll1 = roll1[:,:len_roll2]
    else:
        roll1 = np.concatenate([roll1,np.zeros([roll1.shape[0],len_roll2-len_roll1])],axis=1)
        # roll2 = roll2[:,:len_roll1]
    return roll1,roll2




def get_name_from_maps(filename):
    name = filename.split('-')[1:]
    name = '-'.join(name)
    name = name.split('_')[:-1]
    name = '_'.join(name)
    return name

def str_to_bar_beat(string):
    if '.' in string:
        str_split = string.split('.')
        bar = int(str_split[0])
        beat = int(str_split[1])
        output = [bar,beat,0]

        if len(str_split)>2:
            sub_beat=int(str_split[2])
            output[2] = sub_beat
    else:
        output = [int(string),0,0]
    return output

def get_time(data,bar,beat,sub_beat):

    first_sig = data.time_signature_changes[0]
    if first_sig.numerator == 1 and first_sig.denominator == 4:
        bar += 1

    PPQ = data.resolution
    downbeats = data.get_downbeats()
    try:
        bar_t = downbeats[bar]
    except IndexError as e:
        if bar == len(downbeats):
            # Instead of the first beat of the bar after the last one (that doesn't exist),
            # We take the before-last beat of the last bar
            bar_t = downbeats[-1]
            beat = 'last'
        else:
            raise e


    time_sigs = data.time_signature_changes
    last_sig = True
    for i, sig in enumerate(time_sigs):
        if sig.time > bar_t:
            last_sig = False
            break

    if last_sig:
        current_sig = time_sigs[i]
    else:
        current_sig = time_sigs[i-1]

    if beat == 'last':
        beat = current_sig.numerator - 1

    try:
        assert beat < current_sig.numerator
    except AssertionError:
        print(downbeats)
        for sig in time_sigs:
            print(sig)
        print('-----------')
        print(bar,beat, bar_t)
        print(current_sig)
        raise AssertionError

    beat_ticks = PPQ * 4 / current_sig.denominator
    tick = data.time_to_tick(bar_t) + beat * beat_ticks + sub_beat*beat_ticks/2
    if tick != int(tick):
        print(bar,beat,sub_beat)
        print(current_sig)
        print(tick)
        raise TypeError('Tick is not int!!!')
    else:
        tick = int(tick)
    time =data.tick_to_time(tick)

    return time


# def add_sustain(midi_data):
#     # Turn sustain pedal into actual note lengths
#     new_midi = pm.PrettyMIDI(resolution=480)
#     all_events = []
#     for instr in midi_data.instruments:
#         new_instr = pm.Instrument(program=pm.instrument_name_to_program('Acoustic Grand Piano'),name=instr.name)
#         for note in instr.notes:



def cut_midi(midi_data,start,end):

    new_midi = pm.PrettyMIDI(resolution=480)
    for instr in midi_data.instruments:
        new_instr = pm.Instrument(program=pm.instrument_name_to_program('Acoustic Grand Piano'),name=instr.name)
        for note in instr.notes:
            add_note=False
            if note.start<=start:
                if note.end > start:
                    # Note starts before start and ends after start, there is overlap
                    add_note=True
                else:
                    #Note starts and ends before start, no overlap
                    pass
            else:
                if note.start < end:
                    # Note starts between start and end, regardless of end there is overlap
                    add_note=True
                else:
                    # Note starts after end, no overlap
                    pass

            if add_note:
                new_note = pm.Note(note.velocity,note.pitch,max(0,note.start-start),min(end,note.end-start))
                new_instr.notes.append(new_note)

        ccs_sorted = sorted(instr.control_changes,key=lambda x: x.time)
        cc64_on = False
        for cc in ccs_sorted:
            #Only keep sustain pedal
            if cc.number == 64:
                # Check if CC64 was on before start
                if cc.time < start:
                    cc64_on = cc.value > 64
                elif cc.time>= end:
                    break
                else:
                    if cc.time != start and cc64_on:
                        # Add extra CC to put sustain on
                        new_cc = pm.ControlChange(cc.number,cc.value,0)
                        new_instr.control_changes.append(new_cc)
                        # Add extra CC just the first time
                        cc64_on = False
                    new_cc = pm.ControlChange(cc.number,cc.value,cc.time-start)
                    new_instr.control_changes.append(new_cc)

        new_midi.instruments.append(new_instr)
    return new_midi

def filter_short_gaps(data,thresh=1):
    #Removes all gaps shorter than thresh
    #thresh is in number of steps

    data = 1 - data
    data_filt = filter_short_notes(data,thresh)
    data_filt = 1-data_filt

    return data_filt

def filter_short_notes(data,thresh=1):
    #Removes all notes shorter than thresh
    #thresh is in number of steps
    data_extended = np.pad(data,((0,0),(1,1)),'constant')
    diff = data_extended[:,1:] - data_extended[:,:-1]

    onsets= np.where(diff==1)
    offsets= np.where(diff==-1)

    mask = offsets[1]-onsets[1]>thresh
    onsets_filt = (onsets[0][mask],onsets[1][mask])
    offsets_filt = (offsets[0][mask],offsets[1][mask])

    diff_filtered=np.zeros(data_extended.shape)

    diff_filtered[onsets_filt]=1
    diff_filtered[offsets_filt]=-1

    return np.cumsum(diff_filtered,axis=1)[:,:-2].astype(int)

def get_notes_intervals(data,fs,times=None):
    #Returns the list of note events from a piano-roll

    data_extended = np.pad(data,((0,0),(1,1)),'constant')
    diff = data_extended[:,1:] - data_extended[:,:-1]

    #Onset: when a new note activates (doesn't count repeated notes)
    onsets= np.where(diff==1)
    #Onset: when a new note deactivates (doesn't count repeated notes)
    offsets= np.where(diff==-1)

    assert onsets[0].shape == offsets[0].shape
    assert onsets[1].shape == offsets[1].shape

    pitches = []
    intervals = []
    for [pitch1,onset], [pitch2,offset] in zip(zip(onsets[0],onsets[1]),zip(offsets[0],offsets[1])):
        # print pitch1, pitch2
        # print onset, offset
        assert pitch1 == pitch2
        # Add +1 because pitches cannot be equal to zeros for evaluation
        pitches += [pitch1+1]
        if fs is None:
            intervals += [[onset, offset]]
            if times is not None:
                intervals += [[times[onset], times[min(offset,len(times)-1)]]]
        else:
            intervals += [[onset/float(fs), offset/float(fs)]]
        # print pitches
        # print intervals
    return np.array(pitches), np.array(intervals)


def make_midi_from_roll(roll,fs,note_range=[0,128],times=None):
    #Outputs the waveform corresponding to the pianoroll

    pitches, intervals = get_notes_intervals(roll,fs,times)

    midi_data = pm.PrettyMIDI()
    piano_program = pm.instrument_name_to_program('Acoustic Grand Piano')
    piano = pm.Instrument(program=piano_program)

    for note,(start,end) in zip(pitches,intervals):
        note = pm.Note(
            velocity=100, pitch=note+note_range[0]-1, start=start, end=end) #Note -1 because get_notes_intervals adds +1
        piano.notes.append(note)
    midi_data.instruments.append(piano)
    return midi_data


def save_midi(midi,dest):
    midi.write(dest)

def synthesize_midi(midi):
    # Requires fluidsynth and pyFluidSynth installed!!!
    return midi.fluidsynth(sf2_path="data/YDP-GrandPiano-SF2-20160804/YDP-GrandPiano-20160804.sf2")


def write_sound(sound,filename):

    peak_normalized_audio = pyln.normalize.peak(sound, -1.0)

    # measure the loudness first
    meter = pyln.Meter(44100) # create BS.1770 meter
    loudness = meter.integrated_loudness(sound)

    # loudness normalize audio to -12 dB LUFS
    loudness_normalized_audio = pyln.normalize.loudness(sound, loudness, -12.0)

    sound = 16000*sound #increase gain

    wave_write = wave.open(filename,'w')
    wave_write.setparams([1,2,44100,10,'NONE','noncompressed'])
    ssignal = ''
    for i in range(len(sound)):
       ssignal += wave.struct.pack('h',sound[i]) # transform to binary
    wave_write.writeframes(ssignal)
    wave_write.close()




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

# input_folder = '../MLM_decoding/data/outputs_default_config_split/test'
# dest_folder = 'data/kelz_outputs'
#
# for filename in os.listdir(input_folder):
#     if filename.endswith('.csv') and not filename.startswith('.'):
#         csv_filename = os.path.join(input_folder,filename)
#
#         roll = np.transpose(np.loadtxt(csv_filename),[1,0])
#         roll_bin = (roll>0.5).astype(int)
#
#
#         roll_filt = filter_short_gaps(roll_bin,thresh=3)
#         roll_filt = filter_short_notes(roll_filt,thresh=3)
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
# #### Prepare Li Su MIDI files
# ##############################################################

# input_folder = 'data/lisu_csv'
# dest_folder = 'data/lisu_outputs'
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
#             note = pm.Note(
#                 velocity=100, pitch=int(round(pm.hz_to_note_number(freq))), start=start, end=end)
#             piano.notes.append(note)
#         midi_data.instruments.append(piano)
#
#         dest_filename = os.path.join(dest_folder,filename).replace('.csv','.mid')
#         midi_data.write(dest_filename)




# ##############################################################
# #### Verify segments
# ##############################################################

# MAPS_folder = "data/MAPS_wav"
# AMAPS_folder = "data/A-MAPS_1.2"
# PM_folder = "../MLM_decoding/data/piano-midi-ttv-20p/test"
# csv_folder = 'data/cut_points'
#
# fd = open("data/data_OK.txt", "r")
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
#         for start_str, end_str in cut_points:
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
#                 print(filename,start_str,end_str ,F_measure)
#                 print(pm_start_t, pm_end_t)
#                 print(amaps_start_t, amaps_end_t)
#                 fig,[ax1,ax2,ax3] = plt.subplots(3,1)
#                 ax1.imshow(pm_roll,origin='lower',aspect='auto')
#                 ax2.imshow(amaps_roll,origin='lower',aspect='auto')
#                 ax3.imshow(amaps_roll-pm_roll,origin='lower',aspect='auto',cmap=plt.get_cmap('seismic'))
#                 plt.show()
#         f=open("data/data_OK.txt", "a+")
#         f.write(filename+'\n')

##############################################################
#### Cut A-MAPS MIDI files into segments
##############################################################
#
# MAPS_folder = "data/MAPS_wav"
# AMAPS_folder = "data/A-MAPS_1.2"
# MIDI_input_folders = ["data/lisu_outputs"]#,"data/onsets_and_frames_outputs"]
# MIDI_names = ["lisu"]#,"google"]
# csv_folder = 'data/cut_points'
# dest_folder = 'data/all_midi_cut'
#
# write_AMAPS = False
#
# for filename in os.listdir(MAPS_folder):
#     if filename.endswith('.wav') and not filename.startswith('.') and "chpn-e01" not in filename:# and 'MAPS_MUS-pathetique_2_ENSTDkAm' in filename:
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
#             if write_AMAPS:
#                 amaps_data_cut = cut_midi(amaps_data,start_t,end_t)
#                 amaps_data_cut.write(os.path.join(save_folder,'target.mid'))
#
#             for midi,name in zip(midis,MIDI_names):
#                 midi_cut = cut_midi(midi,start_t,end_t)
#                 midi_cut.write(os.path.join(save_folder,name+'.mid'))
#
#             f= open(os.path.join(save_folder,"duration.txt"),"w+")
#             f.write(str(end_t-start_t))
#             f.close()



# ##############################################################
# #### Convert MIDI files into mp3 files
# ##############################################################

midi_folder = 'data/all_midi_cut'
dest_folder = 'data/all_mp3_cut'
csv_folder = 'data/cut_points'
AMAPS_folder = "data/A-MAPS_1.2"

for subfolder_name in os.listdir(midi_folder):
    subfolder = os.path.join(midi_folder,subfolder_name)
    if os.path.isdir(subfolder):# and 'MAPS_MUS-mz_331_2_ENSTDkCl_16' in subfolder:
        print(subfolder)
        dest_subfolder = os.path.join(dest_folder,subfolder_name)
        safe_mkdir(dest_subfolder)

        # #Retrieve duration of example
        f=open(os.path.join(subfolder,"duration.txt"), "r")
        dur_str = f.read()
        duration = float(dur_str)

        for midi_file in os.listdir(subfolder):
            if midi_file.endswith('.mid') and not midi_file.startswith('.'):
                wav_path = os.path.join(dest_subfolder,midi_file.replace('.mid','.wav'))
                if not os.path.exists(wav_path.replace('.wav','.mp3')):

                    data = pm.PrettyMIDI(os.path.join(subfolder,midi_file))
                    # print midi_file
                    # for instr in data.instruments:
                    #     print instr.control_changes

                    sound1 = synthesize_midi(data)
                    sound1_trim = sound1[:int(duration*44100)]
                    wav_path = os.path.join(dest_subfolder,midi_file.replace('.mid','.wav'))
                    write_sound(sound1_trim,wav_path)

                    sound2 = pydub.AudioSegment.from_wav(wav_path)
                    sound2.export(wav_path.replace('.wav','.mp3'), format="mp3", bitrate="320k")
                    os.remove(wav_path)
