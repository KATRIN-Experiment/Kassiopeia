Directory structure and environmental variables
===============================================

After the compilation is completed and *Kassiopeia* has been installed to the installation directory, it is useful to
set up some environment variables that allow you ton run ``Kassiopeia`` and other commands from any location. A script
is provided that provides a similar functionality to the ``thisroot.sh`` script explained above. To set up *Kassiopeia*
with the script, copy the following lines to your ``~/.bashrc`` (or similar), then logout and login again:

.. code-block:: bash

    #Set up the Kassiopeia environmental variables
    source ~/kassiopeia/install/bin/kasperenv.sh

The script will define a few environment variables that can be used outside of *Kassiopeia*:

- KASPERSYS - the location of *Kassiopeia* binaries, libraries and configuration files.
- KEMFIELD_CACHE - the location of the *KEMField* cache directory
- KASPER_SOURCE - the location of the *Kassiopeia* source directory
- KASPER_INSTALL - the location of the *Kassiopeia* installation directory

The ``KASPERSYS`` and ``KEMFIELD_CACHE`` can, in principle, be changed to different locations before running
simulations. This is intended to allow more flexible configurations on multi-user systems, or when multiple independent
instances of the *Kassiopeia* software are installed. For the typical user, the variables can be left as they are.


The complete set of *Kassiopiea* executables and configuration files will be found in the specified
installation directory. The installation directory is broken down into several components, these are:

- bin
- cache
- config
- data
- doc
- include
- lib
- log
- output
- scratch

The *Kassiopeia* executable can be found under the ``bin`` directory. Also in this directory is the script
``kasperenv.sh`` that was mentioned above.

The ``bin`` directory also contains other executables useful for interacting with the sub-components of *Kassiopeia*
such as the *KEMField* or *KGeoBag* libraries. This included tools for generating particles without running a full
simulation, for calculating electromagnetic fields, or for visualizing the simulation geometry.

The ``lib`` directory contains all of the compiled libraries, as well as cmake and pkgconfig modules to enable linking
against *Kassiopeia* by external programs. The ``include`` directory contains all of the header files of the compiled
programs and libraries.

The other directories: ``cache``, ``config``, ``data``, ``doc``, ``log``, ``output``, and ``scratch`` are all further
sub-divided into parts which relate to each sub-module of the code: *Kassiopeia*, *Kommon*, *KGeoBag*, or *KEMField*.
The ``cache`` and ``scratch`` directories are responsible for storing temporary files needed during run time for later
reuse. The ``data`` directory contains raw data distributed with *Kassiopeia* needed for certain calculations (e.g.
molecular hydrogen scattering cross sections). The ``log`` directory provides space to collect logging output from
simulations, while the ``output`` directory is where simulation output is saved, unless otherwise specified.

