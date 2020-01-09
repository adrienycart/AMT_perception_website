import numpy as np


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

def make_note_index_matrix(notes,intervals,fs=100):
    end_time = np.max(intervals[:,1])
    # Allocate a matrix of zeros - we will add in as we go
    matrix = -np.ones((128, int(fs*end_time)))
    # Make a piano-roll-like matrix holding the index of each note.
    # -1 indicates no note
    for i,(note,interval) in enumerate(zip(notes,intervals)):
        matrix[note,int(interval[0]*fs):int(interval[1]*fs)] = i
    return matrix

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

def get_loudness(midi_pitch, velocity, time):
    # compute decay_rate according to midipitch and note velocity
    time = min(time, 1)
    decay_rate = 0.0 + 0.0 * midi_pitch + 0.0 * velocity
    loudness = velocity * np.exp(-1.0 * decay_rate * time)
    return loudness