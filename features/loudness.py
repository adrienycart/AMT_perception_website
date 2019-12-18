import numpy as np
from utils import precision, recall, Fmeasure, make_note_index_matrix, even_up_rolls, get_decay_rate

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


def loudness_ratio_false_negative(notes_output, intervals_output, notes_target, intervals_target, vel_target, match, min_dur=0.05):
    # loudness ratio of false negative
    # ratio = FN velocity / max loudness in ground truth at onset

    if len(match) == 0:
        return 0.0

    # compute false negatives
    matched_targets, matched_outputs = zip(*match)
    ummatched_targets = list(set(range(len(notes_target))) - set(matched_targets))

    fs = 100

    # get masks for ground truth and false negatives
    target_refs = make_note_index_matrix(notes_target, intervals_target)
    ummatched_target_refs = make_note_index_matrix(notes_output, intervals_output)
    target_refs, output_refs = even_up_rolls(target_refs, output_refs, pad_value = -1)

    roll_target = (target_refs != -1).astype(int)
    roll_output = (output_refs != -1).astype(int)
