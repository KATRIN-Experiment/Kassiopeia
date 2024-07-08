Tools
=====

Some helper tools are provided with *Kassiopeia* that can be used together with simulations:

* ``ROOTFileMerge`` combines several simulation output files into a single file with all runs/events combined. This is
  useful in the case of Monte-Carlo simulations, where *Kassiopeia* is executed several times with the same or slightly
  different settings. Although the individual output files could be analyzed separately, sometimes it is beneficial to
  combine all results into a single file that can then be analyzed in a single go. The program simply takes a list
  of input files, followed by the name of an output file that will be created.
* ``ParticleGenerator`` provides a quick method to generate particles as specified in a configuration file, without
  running a simulations. This is useful for the design of simulations, and to compare and validate the generators used
  by the simulations. The program generates an output text file that contains one line for each generated particle,
  with information about its position, energy, and so on.

All listed programs will show a brief usage summary if called without arguments.
