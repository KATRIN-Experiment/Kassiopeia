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

# retrieve list of valid step ranges
step_presence = reader.getTree('component_step_cell_PRESENCE')
step_valid = zip(*[step_presence['INDEX'], step_presence['LENGTH']])

# retrieve list of step ranges per track
track_step_index = list(zip(*[reader.getTracks('FIRST_STEP_INDEX'), reader.getTracks('LAST_STEP_INDEX')]))
#print("#tracks:", len(track_step_index))

# load step-data tree
reader.loadTree('component_step_cell')
#print('#steps: ', len(reader))
#print('fields: ', dir(reader))

# select a single output field
reader.select('orbital_magnetic_moment')

# iterate over track indices
for first_step_index, last_step_index in track_step_index:

    max_moment = -np.inf
    min_moment = np.inf

    # iterate over steps in given range
    for step in iter(reader):

        step_index = reader.iev - 1

        # advance iterator to first step in this track
        if step_index < first_step_index:
            continue

        # only process steps in valid range within this track
        for first_valid,valid_length in step_valid:

            last_valid = first_valid + valid_length - 1
            if step_index >= first_valid and step_index <= last_valid:

                moment = float(step.orbital_magnetic_moment)
                if moment > max_moment:
                    max_moment = moment
                if moment < min_moment:
                    min_moment = moment

            if first_valid > first_step_index:
                break

        # stop iteration at the end of this track
        if step_index >= last_step_index:
             break

    # calculate result for current track
    deviation = 2.0 * (max_moment - min_moment) / (max_moment + min_moment)
    print("extrema for track <{:g}>".format(deviation))

    break
