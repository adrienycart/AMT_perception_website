import pyloudnorm as pyln
import numpy as np
import pretty_midi as pm
import wave
import os

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
                new_note = pm.Note(note.velocity,note.pitch,max(0,note.start-start),min(end-start,note.end-start))
                if (new_note.start == 0 and new_note.end-new_note.start < 0.05) \
                or (new_note.end == end-start and new_note.end-new_note.start < 0.05):
                    # Do not add short notes at start or end of section
                    # that are due to imprecision in cutting.
                    pass
                else:
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

def apply_sustain_control_changes(midi):
    all_CCs = []
    for instr in midi.instruments:
        all_CCs += instr.control_changes
    all_pedals = [cc for cc in all_CCs if cc.number==64]
    pedals_sorted = sorted(all_pedals,key=lambda x: x.time)

    #Add an extra pedal off at the end, just in case
    pedals_sorted += [pm.ControlChange(64,0,midi.get_end_time())]

    #Create a pedal_ON array such that pedal_ON[i]>0 iff pedal is on at tick i
    #If pedal_ON[i]>0, its value is the time at which pedal becomes off again
    pedal_ON = np.zeros(midi._PrettyMIDI__tick_to_time.shape,dtype=float)
    # -1 if pedal is currently off, otherwise tick time of first time it is on.
    ON_idx = -1
    for cc in pedals_sorted:
        if cc.value > 64:
            if ON_idx < 0:
                ON_idx = midi.time_to_tick(cc.time)
            else:
                # Pedal is already ON
                pass
        else:
            if ON_idx>0:
                pedal_ON[ON_idx:midi.time_to_tick(cc.time)]=cc.time
                ON_idx = -1
            else:
                # Pedal is already OFF
                pass

    # Copy to keep time signatures and tempo changes, but remove notes and CCs
    new_midi = copy.deepcopy(midi)
    new_midi.instruments = []


    # Store the notes per pitch, to trim them afterwards.
    all_notes = np.empty([128],dtype=object)
    for i in range(128):
        all_notes[i] = []


    for instr in midi.instruments:

        # First, extend all the notes until the pedal is off
        for note in instr.notes:
            start_tick = midi.time_to_tick(note.start)
            end_tick = midi.time_to_tick(note.end)

            if np.any(pedal_ON[start_tick:end_tick]>0):
                # Pedal is on while note is on
                end_pedal = np.max(pedal_ON[start_tick:end_tick])
                note.end = max(note.end,end_pedal)
            else:
                # Pedal is not on while note is on, no modifications needed
                pass
            all_notes[note.pitch] += [note]

    new_instr = pm.Instrument(program=pm.instrument_name_to_program('Acoustic Grand Piano'),name="Piano")

    # Then, trim notes so they don't overlap
    for note_list in all_notes:
        if note_list != []:
            note_list = sorted(note_list,key=lambda x: x.start)
            for note_1,note_2 in zip(note_list[:-1],note_list[1:]):
                note_1.end = min(note_1.end,note_2.start)
                new_instr.notes += [note_1]
            new_instr.notes += [note_list[-1]]

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
    return midi.fluidsynth(sf2_path="app/static/data/YDP-GrandPiano-SF2-20160804/YDP-GrandPiano-20160804.sf2")


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


def make_roll(intervals,pitches,shape,fs=100):
    roll = np.zeros(shape)
    for pitch, (start,end) in zip(pitches,intervals):
        # use int() instead of int(round()) to be coherent with PrettyMIDI.get_piano_roll()
        start = int(start*fs)
        end = int(end*fs)
        if start == end:
            end = start+1
        roll[int(note.pitch),start:end]=1
    return roll
