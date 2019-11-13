import numpy as np




##############################################
#### Framewise
##############################################


def specific_pitch_framewise(output,target,fs,n_semitones,down_only=False,delta=0.05):

    FPs = np.logical_and(output == 1, target == 0)

    target_shift_down = np.concatenate([target[n_semitones:,:],np.zeros([n_semitones,target.shape[1]])],axis=0)
    target_shift_up = np.concatenate([np.zeros([n_semitones,target.shape[1]]),target[:-n_semitones,:]],axis=0)

    match_down = FPs*target_shift_up # correspond to when an output is matched with a target n_semitones below
    match_up = FPs*target_shift_down # correspond to when an output is matched with a target n_semitones above

    delta_steps = int(round(delta*fs))
    continuation_mask = np.concatenate([np.concatenate([target[:,i:],np.zeros([target.shape[0],i])],,axis=1)[:,:,None] for i in range(delta_steps)],axis=2)

    match_past = FPs*np.all(continuation_mask==0,axis=2).astype(int)

    if down_only:
        n_match = np.sum(match_down*match_past)
    else:
        match_pitch = np.logical_or(match_down,match_down).astype(int)
        n_match = np.sum(match_pitch*match_past)

    n_match = float(n_match)
    n_FP = np.sum(FPs)
    n_tot = np.sum(output)

    return n_match/n_FP, n_match/n_tot
