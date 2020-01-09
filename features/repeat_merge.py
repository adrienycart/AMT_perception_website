import numpy as np
from utils import precision, recall, Fmeasure, make_note_index_matrix, even_up_rolls



########################################
### Repeated/merged notes
########################################

def repeated_notes(notes_output,intervals_output,notes_target,intervals_target,match,tol=0.8):
    # Here, any note that is a false positive an overlaps with a ground truth note for more
    # than tol percent of its duration is considered a repeated note

    if len(match) == 0:
        return 0.0, 0.0
    else:
        fs = 500

        matched_targets, matched_outputs = zip(*match)
        matched_targets = np.array(matched_targets)
        matched_outputs = list(matched_outputs)
        unmatched_outputs= list(set(range(len(notes_output)))-set(matched_outputs))

        if len(unmatched_outputs)==0:
            # No false positives, return zero
            return 0.0,0.0
        else:

            repeated = []

            target_refs = make_note_index_matrix(notes_target,intervals_target,fs)
            output_refs = make_note_index_matrix(notes_output,intervals_output,fs)
            target_refs,output_refs = even_up_rolls(target_refs,output_refs,pad_value=-1)

            roll_target = (target_refs!=-1).astype(int)
            roll_output = (output_refs!=-1).astype(int)

            if len(unmatched_outputs) == 0:
                return 0.0,0.0
            else:
                for idx in unmatched_outputs:
                    roll_idx = output_refs==idx
                    overlap = np.sum(roll_target[roll_idx])/float(fs)
                    note_duration = intervals_output[idx][1] -  intervals_output[idx][0]
                    if overlap/note_duration > tol:
                        repeated += [idx]


                n_repeat = float(len(repeated))
                tot_err = len(unmatched_outputs)
                tot_notes = len(notes_output)

                return n_repeat/tot_err, n_repeat/tot_notes

def merged_notes(notes_output,intervals_output,notes_target,intervals_target,match,tol=0.8):
    # Here, any note that is a false positive an overlaps with a ground truth note for more
    # than tol percent of its duration is considered a repeated note

    if len(match) == 0:
        return 0.0, 0.0
    else:
        fs = 500

        matched_targets, matched_outputs = zip(*match)
        matched_targets = np.array(matched_targets)
        unmatched_targets= list(set(range(len(notes_target)))-set(matched_targets))

        if len(unmatched_targets)==0:
            # No false positives, return zero
            return 0.0,0.0
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


            return n_repeat/tot_err, n_repeat/tot_notes


##################################
#### OLD STUFF
##################################

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
