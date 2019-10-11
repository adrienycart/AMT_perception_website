import os
import numpy as np
import pretty_midi as pm
import mir_eval



def even_up_rolls(roll1,roll2,pad_value=0):
    #Makes roll1 and roll2 of same size.
    len_roll1 = roll1.shape[1]
    len_roll2 = roll2.shape[1]
    if len_roll1 > len_roll2:
        roll2 = np.concatenate([roll2,pad_value*np.ones([roll2.shape[0],len_roll1-len_roll2])],axis=1)
    else:
        roll1 = np.concatenate([roll1,pad_value*np.ones([roll1.shape[0],len_roll2-len_roll1])],axis=1)
    return roll1,roll2

def get_notes_intervals(midi_data,with_vel=False):
    notes= []
    intervals = []
    if with_vel:
        vels = []

    for instr in midi_data.instruments:
        for note in instr.notes:
            notes += [note.pitch]
            intervals += [[note.start,note.end]]
            if with_vel:
                vels += [note.velocity]
    output = [np.array(notes), np.array(intervals)]
    if with_vel:
        output += [np.array(vels)]

    return output




##################################################
### FRAMEWISE METRICS
##################################################

def TP(data,target):
    return np.sum(np.logical_and(data == 1, target == 1))

def FP(data,target):
    return np.sum(np.logical_and(data == 1, target == 0))

def FN(data,target):
    return np.sum(np.logical_and(data == 0, target == 1))

def precision(tp,fp):
    #Compute precision for  one file
    pre = tp/(tp+fp+np.finfo(float).eps)

    return pre

def recall(tp,fn):
    #Compute recall for  one file
    rec = tp/(tp+fn+np.finfo(float).eps)
    return rec


def accuracy(tp,fp,fn):
    #Compute accuracy for one file
    acc = tp/(tp+fp+fn+np.finfo(float).eps)
    return acc

def Fmeasure(tp,fp,fn):
    #Compute F-measure  one file
    prec = precision(tp,fp)
    rec = recall(tp,fn)
    return 2*prec*rec/(prec+rec+np.finfo(float).eps)


def framewise(output,target):
    tp,fp,fn = TP(output, target),FP(output, target),FN(output, target),

    P_f,R_f,F_f = precision(tp,fp), recall(tp,fn), Fmeasure(tp,fp,fn)
    return P_f,R_f,F_f


##############################################
#### Framewise highest and lowest voice
##############################################

def get_highest(roll):
    # when no note, returns -1
    highest = np.argmax(roll[::-1,:],axis=0)
    highest[highest!=0] = roll.shape[0]-1-highest[highest!=0]
    highest[highest==0] = -1
    return highest

def get_lowest(roll):
    # when no note, returns roll.shape[0]
    lowest = np.argmax(roll,axis=0)
    lowest[lowest==0]= target.shape[0]
    return lowest

def framewise_highest(output, target):

    highest = get_highest(target)

    highest_nonzero = highest[highest!=-1]
    frames_nonzero = np.arange(len(highest))[highest!=-1]

    tp = np.sum(output[highest_nonzero,frames_nonzero])

    fn = np.sum(output[highest_nonzero,frames_nonzero]==0)



    i,j = np.indices(target.shape)
    mask = [i>highest]
    ### Count all false positives above highest reference note)
    fp = np.sum(output[mask])

    return precision(tp,fp),recall(tp, fn), Fmeasure(tp,fp,fn)


def framewise_lowest(output, target):

    lowest = get_lowest(target)

    lowest_nonzero = lowest[lowest!=target.shape[0]]
    frames_nonzero = np.arange(len(lowest))[lowest!=target.shape[0]]

    tp = np.sum(output[lowest_nonzero,frames_nonzero])

    fn = np.sum(output[lowest_nonzero,frames_nonzero]==0)



    i,j = np.indices(target.shape)
    mask = [i<lowest]
    ### Count all false positives above highest reference note)
    fp = np.sum(output[mask])

    return precision(tp,fp),recall(tp, fn), Fmeasure(tp,fp,fn)

########################################
### Notewise highest and lowest voice
########################################

def make_note_index_matrix(notes,intervals):
    fs = 100
    end_time = np.max(intervals[:,1])
    # Allocate a matrix of zeros - we will add in as we go
    matrix = -np.ones((128, int(fs*end_time)))
    # Make a piano-roll-like matrix holding the index of each note.
    # -1 indicates no note
    for i,(note,interval) in enumerate(zip(notes,intervals)):
        matrix[note,int(interval[0]*fs):int(interval[1]*fs)] = i
    return matrix

def notewise_highest(notes_output,intervals_output,notes_target,intervals_target,match,min_dur=0.05):
    #min_dur represents the minimum duration a note has to be the highest to be considered
    #in the skyline
    fs = 100

    # Get the list of highest notes
    target_refs = make_note_index_matrix(notes_target,intervals_target)
    output_refs = make_note_index_matrix(notes_output,intervals_output)
    target_refs,output_refs = even_up_rolls(target_refs,output_refs,pad_value=-1)

    roll_target = (target_refs!=-1).astype(int)
    roll_output = (output_refs!=-1).astype(int)

    highest = get_highest(roll_target)
    highest_nonzero = highest[highest!=-1]
    frames_nonzero = np.arange(len(highest))[highest!=-1]

    highest_notes_idx, count = np.unique(target_refs[highest_nonzero,frames_nonzero],return_counts=True)
    highest_notes_idx = highest_notes_idx[count/float(fs) > min_dur]

    # Compute true positives
    # NB: matching gives indexes (idx_target,idx_output)

    matched_targets, matched_outputs = zip(*match)
    matched_targets_is_highest = [idx for idx in matched_targets if idx in highest_notes_idx]
    tp = len(matched_targets_is_highest)

    # Compute false negatives
    unmatched_targets= list(set(range(len(notes_target)))-set(matched_targets))
    unmatched_targets_is_highest = [idx for idx in unmatched_targets if idx in highest_notes_idx]
    fn = len(unmatched_targets_is_highest)

    # Compute false positives
    # Count all false positives that are above the highest note
    i,j = np.indices(target_refs.shape)
    higher_mask = [i>highest]
    higher_notes_idx, count = np.unique(output_refs[higher_mask],return_counts=True)
    count = count[higher_notes_idx!= -1]
    higher_notes_idx = higher_notes_idx[higher_notes_idx!= -1]
    print higher_notes_idx, count
    higher_notes_idx = higher_notes_idx[count/float(fs) > min_dur]

    unmatched_outputs= list(set(range(len(notes_output)))-set(matched_outputs))
    unmatched_outputs_is_higher = [idx for idx in unmatched_outputs if idx in higher_notes_idx]
    fp = len(unmatched_outputs_is_higher)

    import matplotlib.pyplot as plt
    fig,((ax0,ax1),(ax2,ax3)) = plt.subplots(2,2)
    ax0.imshow(roll_target,aspect='auto',origin='lower')
    ax1.imshow(roll_output,aspect='auto',origin='lower')
    display1 = np.zeros_like(roll_output)
    display2 = np.zeros_like(roll_output)
    for i in matched_targets:
        display1[notes_target[i],int(intervals_target[i][0]*fs):int(intervals_target[i,1]*fs)] = 1
    for i in unmatched_outputs_is_higher:
        display2[notes_output[i],int(intervals_output[i][0]*fs):int(intervals_output[i,1]*fs)] = 1

    display2[higher_mask]+=3

    ax2.imshow(display1,aspect='auto',origin='lower')
    ax3.imshow(display2,aspect='auto',origin='lower')
    plt.show()











MIDI_path = 'app/static/data/all_midi_cut'
systems = ['kelz','lisu','google','cheng']


for example in os.listdir(MIDI_path)[0:10]:
    example_path = os.path.join(MIDI_path,example)

    print '________________'
    print example

    target_data = pm.PrettyMIDI(os.path.join(example_path,'target.mid'))
    for system in systems:
        print system
        system_data = pm.PrettyMIDI(os.path.join(example_path,system+'.mid'))

        target_pr = (target_data.get_piano_roll()>0).astype(int)
        system_pr = (system_data.get_piano_roll()>0).astype(int)

        target_pr,system_pr= even_up_rolls(target_pr,system_pr)

        P_f,R_f,F_f = framewise(system_pr,target_pr)
        print "Frame P,R,F:", P_f,R_f,F_f

        notes_target, intervals_target = get_notes_intervals(target_data)
        notes_system, intervals_system = get_notes_intervals(system_data)


        match_on = mir_eval.transcription.match_notes(intervals_target, notes_target, intervals_system, notes_system,offset_ratio=None, pitch_tolerance=0.25)
        match_onoff = mir_eval.transcription.match_notes(intervals_target, notes_target, intervals_system, notes_system,offset_ratio=0.2, pitch_tolerance=0.25)

        P_n_on = float(len(match_on))/len(notes_system)
        R_n_on = float(len(match_on))/len(notes_target)
        F_n_on = 2*P_n_on*R_n_on/(P_n_on+R_n_on+np.finfo(float).eps)
        print "Note-On P,R,F:", P_n_on,R_n_on,F_n_on

        P_n_onoff = float(len(match_onoff))/len(notes_system)
        R_n_onoff = float(len(match_onoff))/len(notes_target)
        F_n_onoff = 2*P_n_onoff*R_n_onoff/(P_n_onoff+R_n_onoff+np.finfo(float).eps)
        print "Note-OnOff P,R,F:", P_n_onoff,R_n_onoff,F_n_onoff

        notewise_highest(notes_system,intervals_system,notes_target,intervals_target,match_on)

        # print match




############
##### TESTS
############