import os
import csv
import pretty_midi as pm
import numpy as np
import matplotlib.pyplot as plt
from features.utils import get_time

cut_points_path = "app\\static\\data\\cut_points"
all_midi_path = "app\\static\\data\\all_midi\\A-MAPS_1.2_with_pedal"

MIDI_path = "app/static/data/all_midi_cut"

for folder in os.listdir(MIDI_path):
    s = len("MAPS_MUS-")
    e = folder.index("_ENS")
    print(folder)
    print(folder[s:e])


# get cut points
cut_points_dict = dict()
for filename in os.listdir(cut_points_path):
    musicname = filename[:-4]
    cut_points_dict[musicname] = np.genfromtxt(os.path.join(cut_points_path, filename), dtype='str')

for filename in [os.path.join(all_midi_path, file) for file in os.listdir(all_midi_path)]:

    midi_data = pm.PrettyMIDI(filename)
    PPQ = midi_data.resolution
    end_tick = midi_data.time_to_tick(midi_data.get_end_time())
    ticks = np.arange(0, end_tick, PPQ/4)
    quarter_times = np.array([midi_data.tick_to_time(t) for t in ticks])

    plt.plot(quarter_times[1:] - quarter_times[:-1])
    plt.show()

