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

def make_note_index_matrix(notes,intervals,fs=100):
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
    if len(match) == 0:
        return 0.0, 0.0, 0.0
    else:

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
        higher_notes_idx = higher_notes_idx[count/float(fs) > min_dur]

        unmatched_outputs= list(set(range(len(notes_output)))-set(matched_outputs))
        unmatched_outputs_is_higher = [idx for idx in unmatched_outputs if idx in higher_notes_idx]
        fp = len(unmatched_outputs_is_higher)

        # import matplotlib.pyplot as plt
        # fig,((ax0,ax1),(ax2,ax3)) = plt.subplots(2,2)
        # ax0.imshow(roll_target,aspect='auto',origin='lower')
        # ax1.imshow(roll_output,aspect='auto',origin='lower')
        # display1 = np.zeros_like(roll_output)
        # display2 = np.zeros_like(roll_output)
        # for i in matched_targets:
        #     display1[notes_target[i],int(intervals_target[i][0]*fs):int(intervals_target[i,1]*fs)] = 1
        # for i in unmatched_outputs_is_higher:
        #     display2[notes_output[i],int(intervals_output[i][0]*fs):int(intervals_output[i,1]*fs)] = 1
        #
        # display2[higher_mask]+=3
        #
        # ax2.imshow(display1,aspect='auto',origin='lower')
        # ax3.imshow(display2,aspect='auto',origin='lower')
        # plt.show()

        return precision(tp,fp),recall(tp, fn), Fmeasure(tp,fp,fn)


def notewise_lowest(notes_output,intervals_output,notes_target,intervals_target,match,min_dur=0.05):

    if len(match) == 0:
        return 0.0,0.0,0.0
    else:
        #min_dur represents the minimum duration a note has to be the highest to be considered
        #in the skyline
        fs = 100

        # Get the list of highest notes
        target_refs = make_note_index_matrix(notes_target,intervals_target)
        output_refs = make_note_index_matrix(notes_output,intervals_output)
        target_refs,output_refs = even_up_rolls(target_refs,output_refs,pad_value=-1)

        roll_target = (target_refs!=-1).astype(int)
        roll_output = (output_refs!=-1).astype(int)

        lowest = get_lowest(roll_target)
        lowest_nonzero = lowest[lowest!=-1]
        frames_nonzero = np.arange(len(lowest))[lowest!=-1]

        lowest_notes_idx, count = np.unique(target_refs[lowest_nonzero,frames_nonzero],return_counts=True)
        lowest_notes_idx = lowest_notes_idx[count/float(fs) > min_dur]

        # Compute true positives
        # NB: matching gives indexes (idx_target,idx_output)

        matched_targets, matched_outputs = zip(*match)
        matched_targets_is_lowest = [idx for idx in matched_targets if idx in lowest_notes_idx]
        tp = len(matched_targets_is_lowest)

        # Compute false negatives
        unmatched_targets= list(set(range(len(notes_target)))-set(matched_targets))
        unmatched_targets_is_lowest = [idx for idx in unmatched_targets if idx in lowest_notes_idx]
        fn = len(unmatched_targets_is_lowest)

        # Compute false positives
        # Count all false positives that are above the lowest note
        i,j = np.indices(target_refs.shape)
        lower_mask = [i<lowest]
        lower_notes_idx, count = np.unique(output_refs[lower_mask],return_counts=True)
        count = count[lower_notes_idx!= -1]
        lower_notes_idx = lower_notes_idx[lower_notes_idx!= -1]
        # print(lower_notes_idx, count)
        lower_notes_idx = lower_notes_idx[count/float(fs) > min_dur]

        unmatched_outputs= list(set(range(len(notes_output)))-set(matched_outputs))
        unmatched_outputs_is_lower = [idx for idx in unmatched_outputs if idx in lower_notes_idx]
        fp = len(unmatched_outputs_is_lower)

        # import matplotlib.pyplot as plt
        # fig,((ax0,ax1),(ax2,ax3)) = plt.subplots(2,2)
        # ax0.imshow(roll_target,aspect='auto',origin='lower')
        # ax1.imshow(roll_output,aspect='auto',origin='lower')
        # display1 = np.zeros_like(roll_output)
        # display2 = np.zeros_like(roll_output)
        # for i in matched_targets:
        #     display1[notes_target[i],int(intervals_target[i][0]*fs):int(intervals_target[i,1]*fs)] = 1
        # for i in unmatched_outputs_is_lower:
        #     display2[notes_output[i],int(intervals_output[i][0]*fs):int(intervals_output[i,1]*fs)] = 1
        #
        # display2[lower_mask]+=3
        #
        # ax2.imshow(display1,aspect='auto',origin='lower')
        # ax3.imshow(display2,aspect='auto',origin='lower')
        # plt.show()

        return precision(tp,fp),recall(tp, fn), Fmeasure(tp,fp,fn)


########################################
### Loudness
########################################

def false_negative_loudness(match,vel_target):

    if len(match) == 0:
        return 128.0
    else:
        matched_targets, matched_outputs = zip(*match)
        unmatched_targets= list(set(range(len(vel_target)))-set(matched_targets))

        avg_vel = np.mean(vel_target)

        if len(unmatched_targets) == 0:
            avg_unmatched = 0
        else:
            unmatched_vels = vel_target[unmatched_targets]
            avg_unmatched = np.mean(unmatched_vels)

        return avg_unmatched / float(avg_vel)


########################################
### Polyphony
########################################

def polyphony_level_seq(roll):
    return np.sum(roll,axis=[0])

def false_negative_polyphony_level(roll_target,intervals_target,match):
    fs = 100

    if len(match) == 0:
        unmatched_targets = list(range(intervals_target))
    else:
        matched_targets, matched_outputs = zip(*match)
        unmatched_targets= list(set(range(len(vel_target)))-set(matched_targets))

    unmatched_intervals = intervals_target[unmatched_targets,:]

    all_avg_poly = []

    for [start,end] in unmatched_intervals:
        start_idx = int(round(start*fs))
        end_idx = int(round(end*fs))
        avg_poly = np.mean(np.sum(roll_target[:,start_idx:end_idx],axis=0))
        all_avg_poly += [avg_poly]

    return []


########################################
### Out-of-key notes
########################################


def make_key_mask(target_roll):

    i = np.arange(target_roll.shape[0])
    output = np.zeros([12],dtype=float)
    for p in range(12):
        active_pitch_class = np.max(target_roll[i%12==p,:],axis=0)
        output[p] = np.mean(active_pitch_class)

    return output

def out_key_errors(notes_output,match,mask):

    if len(match) == 0:
        unmatched_outputs = list(range(notes_output))
    else:
        matched_targets, matched_outputs = zip(*match)
        unmatched_outputs= list(set(range(len(notes_output)))-set(matched_outputs))

    if len(unmatched_outputs) == 0:
        return 0.0,0.0
    else:
        unmatched_weights = []
        for i in unmatched_outputs:
            unmatched_weights += [1- mask[notes_output[i]%12]]

        all_weights = []
        for note in notes_output:
            all_weights += [1- mask[note%12]]

        return np.mean(all_weights), sum(unmatched_weights)/sum(all_weights)

def out_key_errors_binary_mask(notes_output,match,mask,mask_thresh=0.1):


    in_mask = mask>mask_thresh

    # import matplotlib.pyplot as plt
    # fig, (ax0,ax1) = plt.subplots(2,1)
    # ax0.plot(mask)
    # ax1.plot(in_mask.astype(int))
    # plt.show()

    if len(match) == 0:
        unmatched_outputs = list(range(notes_output))
    else:
        matched_targets, matched_outputs = zip(*match)
        unmatched_outputs= list(set(range(len(notes_output)))-set(matched_outputs))

    if len(unmatched_outputs) == 0:
        return 0.0,0.0
    else:
        out_key_unmatched = []
        for i in unmatched_outputs:
            if not in_mask[notes_output[i]%12]:
                out_key_unmatched += [notes_output[i]]

        tot_out_key = float(len(out_key_unmatched))
        tot_err = len(unmatched_outputs)
        tot_notes = len(notes_output)

        return tot_out_key/tot_err, tot_out_key/tot_notes


########################################
### Repeated/merged notes
########################################

def repeated_notes_WITH_CORRECT_ONSET_ONLY(intervals_target,notes_output,intervals_output,match_on):
    # Here, a repeated note is counted if and only if there is a correctly detected note before (onset-only)

    if len(match_on) == 0:
        # No matches, return zero
        return 0.0,0.0
    else:
        matched_targets, matched_outputs = zip(*match_on)
        matched_targets = np.array(matched_targets)
        matched_outputs = list(matched_outputs)
        unmatched_outputs= list(set(range(len(notes_output)))-set(matched_outputs))

        if len(unmatched_outputs)==0:
            # No false positives, return zero
            return 0.0,0.0,[]
        else:

            matched_notes = notes_output[matched_outputs]
            unmatched_notes = notes_output[unmatched_outputs]
            # keys are pitches, values are the target intervals that have been matched to that pitch
            pitch_dict = {}

            print matched_outputs,unmatched_outputs
            for note in unmatched_notes:
                if not note in pitch_dict:
                    match_indices = np.where(matched_notes==note)[0]
                    # print match_indices
                    intervals_indices = matched_targets[match_indices]
                    # print note,intervals_indices,intervals_target[intervals_indices]
                    pitch_dict[note] = intervals_target[intervals_indices]
                else:
                    #Only compute that the first time this pitch is encountered (same results)
                    pass

            repeated = []
            for unmatched_idx in unmatched_outputs:
                note = notes_output[unmatched_idx]
                matching_target_intervals = pitch_dict[note]
                # If there are some matched notes with same pitch
                # print matching_target_intervals
                if not matching_target_intervals.size==0:
                    interval_out = intervals_output[unmatched_idx]
                    # print matching_target_intervals, interval_out
                    overlap = (np.minimum(matching_target_intervals[:,1],interval_out[1]) - np.maximum(matching_target_intervals[:,0],interval_out[0]))/(interval_out[1]-interval_out[0])
                    is_repeated = overlap > 0.8
                    print note, overlap
                    assert sum(is_repeated.astype(int))==1 or sum(is_repeated.astype(int))==0

                    if np.any(is_repeated):
                        repeated += [unmatched_idx]


            n_repeat = float(len(repeated))
            tot_err = len(unmatched_outputs)
            tot_notes = len(notes_output)

            print n_repeat, tot_err, tot_notes

            return n_repeat/tot_err, n_repeat/tot_notes, repeated


def repeated_notes(notes_output,intervals_output,notes_target,intervals_target,match,tol=0.8):
    # Here, any note that is a false positive an overlaps with a ground truth note for more
    # than tol percent of its duration is considered a repeated note

    if len(match) == 0:
        return 0.0, 0.0, 0.0
    else:
        fs = 500

        matched_targets, matched_outputs = zip(*match_on)
        matched_targets = np.array(matched_targets)
        matched_outputs = list(matched_outputs)
        unmatched_outputs= list(set(range(len(notes_output)))-set(matched_outputs))

        if len(unmatched_outputs)==0:
            # No false positives, return zero
            return 0.0,0.0,[]
        else:

            repeated = []

            target_refs = make_note_index_matrix(notes_target,intervals_target,fs)
            output_refs = make_note_index_matrix(notes_output,intervals_output,fs)
            target_refs,output_refs = even_up_rolls(target_refs,output_refs,pad_value=-1)

            roll_target = (target_refs!=-1).astype(int)
            roll_output = (output_refs!=-1).astype(int)

            for idx in unmatched_outputs:
                roll_idx = output_refs==idx
                overlap = np.sum(roll_target[roll_idx])/float(fs)
                note_duration = intervals_output[idx][1] -  intervals_output[idx][0]
                if overlap/note_duration > tol:
                    repeated += [idx]


            n_repeat = float(len(repeated))
            tot_err = len(unmatched_outputs)
            tot_notes = len(notes_output)

            print n_repeat, tot_err, tot_notes

            return n_repeat/tot_err, n_repeat/tot_notes

def merged_notes(notes_output,intervals_output,notes_target,intervals_target,match,tol=0.8):
    # Here, any note that is a false positive an overlaps with a ground truth note for more
    # than tol percent of its duration is considered a repeated note

    if len(match) == 0:
        return 0.0, 0.0, 0.0
    else:
        fs = 500

        matched_targets, matched_outputs = zip(*match_on)
        matched_targets = np.array(matched_targets)
        unmatched_targets= list(set(range(len(notes_target)))-set(matched_targets))

        if len(unmatched_targets)==0:
            # No false positives, return zero
            return 0.0,0.0,[]
        else:

            repeated = []

            target_refs = make_note_index_matrix(notes_target,intervals_target,fs)
            output_refs = make_note_index_matrix(notes_output,intervals_output,fs)
            target_refs,output_refs = even_up_rolls(target_refs,output_refs,pad_value=-1)

            roll_target = (target_refs!=-1).astype(int)
            roll_output = (output_refs!=-1).astype(int)

            for idx in unmatched_targets:
                roll_idx = target_refs==idx
                overlap = np.sum(roll_output[roll_idx])/float(fs)
                note_duration = intervals_target[idx][1] -  intervals_target[idx][0]
                if overlap/note_duration > tol:
                    repeated += [idx]


            n_repeat = float(len(repeated))
            tot_err = len(unmatched_targets)
            tot_notes = len(notes_output)

            print n_repeat, tot_err, tot_notes

            return n_repeat/tot_err, n_repeat/tot_notes
















MIDI_path = 'app/static/data/all_midi_cut'
systems = ['kelz','lisu','google','cheng']


for example in os.listdir(MIDI_path)[0:10]:
    example_path = os.path.join(MIDI_path,example)

    print('________________')
    print(example)

    target_data = pm.PrettyMIDI(os.path.join(example_path,'target.mid'))
    for system in systems:
        # if example=="MAPS_MUS-scn15_11_ENSTDkAm_5" and system=='google':
        if system=='google' or system == 'kelz':
            print(system)
            system_data = pm.PrettyMIDI(os.path.join(example_path,system+'.mid'))

            target_pr = (target_data.get_piano_roll()>0).astype(int)
            system_pr = (system_data.get_piano_roll()>0).astype(int)

            target_pr,system_pr= even_up_rolls(target_pr,system_pr)

            P_f,R_f,F_f = framewise(system_pr,target_pr)
            print("Frame P,R,F:", P_f,R_f,F_f)

            notes_target, intervals_target, vel_target = get_notes_intervals(target_data,with_vel=True)
            notes_system, intervals_system = get_notes_intervals(system_data)


            match_on = mir_eval.transcription.match_notes(intervals_target, notes_target, intervals_system, notes_system,offset_ratio=None, pitch_tolerance=0.25)
            match_onoff = mir_eval.transcription.match_notes(intervals_target, notes_target, intervals_system, notes_system,offset_ratio=0.2, pitch_tolerance=0.25)

            P_n_on = float(len(match_on))/len(notes_system)
            R_n_on = float(len(match_on))/len(notes_target)
            F_n_on = 2*P_n_on*R_n_on/(P_n_on+R_n_on+np.finfo(float).eps)
            print("Note-On P,R,F:", P_n_on,R_n_on,F_n_on)

            P_n_onoff = float(len(match_onoff))/len(notes_system)
            R_n_onoff = float(len(match_onoff))/len(notes_target)
            F_n_onoff = 2*P_n_onoff*R_n_onoff/(P_n_onoff+R_n_onoff+np.finfo(float).eps)
            print("Note-OnOff P,R,F:", P_n_onoff,R_n_onoff,F_n_onoff)

            # notewise_highest(notes_system,intervals_system,notes_target,intervals_target,match_onoff)

            mask = make_key_mask(target_pr)
            _,_,repeated = repeated_notes(intervals_target,notes_system,intervals_system,match_on)

            if len(repeated)>0 or True:
                import matplotlib.pyplot as plt
                fig,((ax0,ax1),(ax2,ax3)) = plt.subplots(2,2)
                for i in range(len(notes_target)):
                    target_pr[notes_target[i],int(intervals_target[i][0]*100)] += 1
                ax0.imshow(target_pr,aspect='auto',origin='lower')
                for i in range(len(notes_system)):
                    if notes_system[i] == 62:
                        print intervals_system[i]
                    system_pr[notes_system[i],int(intervals_system[i][0]*100)] += 1
                ax1.imshow(system_pr,aspect='auto',origin='lower')
                display1 = np.zeros_like(system_pr)
                display2 = np.zeros_like(system_pr)
                matched_targets,matched_outputs = zip(*match_on)
                for i in matched_targets:
                    display1[notes_target[i],int(intervals_target[i][0]*100):int(intervals_target[i,1]*100)] = 1
                    display1[notes_target[i],int(intervals_target[i][0]*100)] += 1
                for i in matched_outputs:
                    display2[notes_system[i],int(intervals_system[i][0]*100):int(intervals_system[i,1]*100)] = 1
                    display2[notes_system[i],int(intervals_system[i][0]*100)] += 1
                for i in repeated:
                    display2[notes_system[i],int(intervals_system[i][0]*100):int(intervals_system[i,1]*100)] = 3
                    display2[notes_system[i],int(intervals_system[i][0]*100)] += 1

                ax2.imshow(display1,aspect='auto',origin='lower')
                ax3.imshow(display2,aspect='auto',origin='lower')
                plt.show()

            # print match




############
##### TESTS
############
