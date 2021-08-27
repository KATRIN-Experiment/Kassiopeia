#!/usr/bin/env python3

import sys
import os

# import module from Kasper
sys.path += [ os.environ['KASPERSYS'] + '/lib/python', os.environ['KASPERSYS'] + '/lib64/python' ]
import KassiopeiaReader

if len(sys.argv) < 2:
    raise RuntimeError("No input file provided.")

# create reader instance
reader = KassiopeiaReader.Iterator(sys.argv[1])

# load step-data tree
reader.loadTree('component_step_cell')
#print("#tracks:", len(reader.getTracks('TRACK_INDEX')))
#print('#steps: ', len(reader))
#print('fields: ', dir(reader))

# select a single output field
reader.select('orbital_magnetic_moment')

# retrieve list of valid step ranges
step_presence = reader.getTree('component_step_cell_PRESENCE')
step_valid = zip(*[step_presence['INDEX'], step_presence['LENGTH']])

# create step iterator
step = iter(reader)
#print('fields: ', dir(step))

# iterate over track indices
for track_index in reader.getTracks('TRACK_INDEX'):

    # retrieve index of first/last step for current track
    first_step_index = reader.getTracks('FIRST_STEP_INDEX')[track_index]
    last_step_index = reader.getTracks('LAST_STEP_INDEX')[track_index]

    # adjust first/last step range to include only valid entries
    for first_valid,valid_length in step_valid:
        last_valid = first_valid + valid_length - 1
        if first_valid >= first_step_index:
            first_step_index = first_valid
            if last_valid < last_step_index:
                last_step_index = last_valid
            break

    #print("track #{} valid interval: {} .. {}".format(track_index, first_step_index, last_step_index))

    # advance iterator to first step
    while reader.iev < first_step_index:
        item = next(step)

    max_moment = -float('Inf')
    min_moment = float('Inf')

    # iterate over steps in given range
    while reader.iev <= last_step_index:

        # advance step iterator
        item = next(step)

        moment = float(item.orbital_magnetic_moment)
        if moment > max_moment:
            max_moment = moment
        if moment < min_moment:
            min_moment = moment

    # calculate result for current track
    deviation = 2.0 * (max_moment - min_moment) / (max_moment + min_moment)
    print("extrema for track <{:g}>".format(deviation))

