import os
import numpy as np
import pretty_midi as pm
import mir_eval
import matplotlib.pyplot as plt
import features.utils as utils
from features.high_low_voice import framewise_highest, framewise_lowest, notewise_highest, notewise_lowest
from features.loudness import false_negative_loudness, loudness_ratio_false_negative
from features.out_key import make_key_mask, out_key_errors, out_key_errors_binary_mask
from features.polyphony import polyphony_level_seq, false_negative_polyphony_level
from features.repeat_merge import repeated_notes, merged_notes
from features.specific_pitch import specific_pitch_framewise, specific_pitch_notewise
from features.rhythm import rhythm_histogram, rhythm_dispersion
import warnings
warnings.filterwarnings("ignore")


def plot_piano_roll(pr):
	fig = plt.figure()
	fig = plt.imshow(pr)
	plt.show()
	return


def test_high_low_voice_framewise(output, target):
	highest_p, highest_r, highest_f = framewise_highest(output, target)
	lowest_p, lowest_r, lowest_f = framewise_lowest(output, target)
	print('highest evaluation: ' + str(highest_p) + ', ' + str(highest_r) + ', ' + str(highest_f))
	print('lowest evaluation: ' + str(lowest_p) + ', ' + str(lowest_r) + ', ' + str(lowest_f))
	return highest_p, highest_r, highest_f, lowest_p, lowest_r, lowest_f


def test_high_low_voice_notewise(notes_output, intervals_output, notes_target, intervals_target, match):
	highest_p, highest_r, highest_f = notewise_highest(notes_output, intervals_output, notes_target, intervals_target, match)
	lowest_p, lowest_r, lowest_f = notewise_lowest(notes_output, intervals_output, notes_target, intervals_target, match)
	print('highest evaluation: ' + str(highest_p) + ', ' + str(highest_r) + ', ' + str(highest_f))
	print('lowest evaluation: ' + str(lowest_p) + ', ' + str(lowest_r) + ', ' + str(lowest_f))
	return highest_p, highest_r, highest_f, lowest_p, lowest_r, lowest_f


def test_loudness(match, vel_target):
	value = false_negative_loudness(match, vel_target)
	print('loudness value: ' + str(value))
	return value


def test_out_key_non_binary(notes_output, match, mask):
	ratio_1, ratio_2 = out_key_errors(notes_output, match, mask)
	print('ratios: ' + str(ratio_1) + ', ' + str(ratio_2))
	return ratio_1, ratio_2


def test_out_key_binary(notes_output, match, mask):
	ratio_1, ratio_2 = out_key_errors_binary_mask(notes_output, match, mask)
	print('ratios: ' + str(ratio_1) + ', ' + str(ratio_2))
	return ratio_1, ratio_2


def test_polyphony(roll_target, intervals_target, match):
	level = polyphony_level_seq(roll_target)
	level_FN = false_negative_polyphony_level(roll_target, intervals_target, match)
	# print('polyphony level: ' + str(level))
	print('polyphony level FN: ' + str(level_FN))
	return level


def test_repeat_merge(notes_output, intervals_output, notes_target, intervals_target, match):
	repeat_ratio = repeated_notes(notes_output, intervals_output, notes_target, intervals_target, match)
	merge_ratio = merged_notes(notes_output, intervals_output, notes_target, intervals_target, match)
	print('repeated notes ratios: ' + str(repeat_ratio[0]) + ', ' + str(repeat_ratio[1]))
	print('merged notes ratios: ' + str(merge_ratio[0]) + ', ' + str(merge_ratio[1]))
	return repeat_ratio, merge_ratio


def test_specific_pitch(output, target, fs):
	semitone = specific_pitch_framewise(output, target, fs, 1)
	octave = specific_pitch_framewise(output, target, fs, 12)
	third_harmonic = specific_pitch_framewise(output, target, fs, 19)
	print('semitone error: ' + str(semitone))
	print('octave error: ' + str(octave))
	print('third_harmonic: ' + str(third_harmonic))
	return semitone, octave, third_harmonic


MIDI_path = 'app/static/data/all_midi_cut'
systems = ['kelz', 'lisu', 'google', 'cheng']
fs = 44100


for example in os.listdir(MIDI_path)[:20]:
	example_path = os.path.join(MIDI_path, example)  # folder path
	print('\n\npath = ' + example_path)
	target_data = pm.PrettyMIDI(os.path.join(example_path, 'target.mid'))

	for system in systems:
		if system == 'google':
			# only test 'google'
			system_data = pm.PrettyMIDI(os.path.join(example_path, system + '.mid'))

			# target and system piano rolls
			# print('getting piano roll...')
			target_pr = (target_data.get_piano_roll()>0).astype(int)
			system_pr = (system_data.get_piano_roll()>0).astype(int)
			target_pr, system_pr = utils.even_up_rolls(target_pr, system_pr)

			# plot_piano_roll(target_pr)
			# plot_piano_roll(system_pr)

			# calculate note pitch, intervals and velocities
			notes_target, intervals_target, vel_target = utils.get_notes_intervals(target_data, with_vel=True)
			notes_system, intervals_system = utils.get_notes_intervals(system_data)

			# calculate true positives
			match_on = mir_eval.transcription.match_notes(intervals_target, notes_target, intervals_system, notes_system, offset_ratio=None, pitch_tolerance=0.25)
			match_onoff = mir_eval.transcription.match_notes(intervals_target, notes_target, intervals_system, notes_system, offset_ratio=0.2, pitch_tolerance=0.25)

			# test features...
			# print('\n test high_low_voice =======================================')
			# print('\n>>>> test high_low_voice framewise')
			# test_high_low_voice_framewise(system_pr, target_pr)

			# print('\n>>>> test high_low_voice notewise, onset only')
			# test_high_low_voice_notewise(notes_system, intervals_system, notes_target, intervals_target, match_on)  # onset only

			# print('\n>>>> test high_low_voice notewise, onset and offset')
			# test_high_low_voice_notewise(notes_system, intervals_system, notes_target, intervals_target, match_onoff)  # onset and offset

			# print('\n test loudness =============================================')
			# print('\n>>>> test loudness, onset only')
			# test_loudness(match_on, vel_target)

			# ratio = loudness_ratio_false_negative(notes_target, intervals_target, vel_target, match_on)
			# print('loudness ratio: ' + str(ratio))

			# print('\n test out_key ===============================================')
			# mask = make_key_mask(target_pr)

			# print('\n>>>> non-binary pitch profile, onset only')
			# test_out_key_non_binary(notes_system, match_on, mask)

			# print('\n>>>> binary pitch profile, onset only')
			# test_out_key_binary(notes_system, match_on, mask)

			# print('\n test polyphony ==============================================')
			# test_polyphony(target_pr, intervals_target, match_on)

			# print('\n test repeat_merge ===========================================')
			# test_repeat_merge(notes_system, intervals_system, notes_target, intervals_target, match_on)

			# print('\n test specific_pitch ===========================================')
			# test_specific_pitch(system_pr, target_pr, fs)
			# r1, r2 = specific_pitch_notewise(notes_system, intervals_system, notes_target, intervals_target, match_on, n_semitones=1)
			# print(str(r1) + "   " + str(r2))

			print('\n test rhythm =====================================================')
			f1, f2 = rhythm_histogram(intervals_system, intervals_target)
			print("logged spectral flatness: " + str(f1) + "(output)   " + str(f2) + "(target)")
			mean_drift, max_drift = rhythm_dispersion(intervals_system, intervals_target)
			print("cluster centre drift: " + str(mean_drift) + "(mean)  " + str(max_drift) + "(max)")




