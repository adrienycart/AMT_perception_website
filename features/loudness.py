import numpy as np


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
