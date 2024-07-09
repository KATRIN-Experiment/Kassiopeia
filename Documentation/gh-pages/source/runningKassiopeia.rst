Running Kassiopeia
==================


After installation, and assuming the proper environmental variables have been set by running the script
``kasperenv.sh``, *Kassiopeia* can be run directly from the command prompt. The script sets the environment variables
accordingly so that the variable ``KASPERSYS`` points to the installation directory, and *Kassiopeia* can be found
in the standard executable path. To run a *Kassiopeia* simulation, the usage is simply:

.. code-block:: bash

    Kassiopeia <path-to-xml-config-file>

*Kassiopeia* also includes advanced variable replacement functionality, which makes it easy to modify one or many
*simulation parameters on-the-fly. This can be done with the following usage:

.. code-block:: bash

    Kassiopeia <path-to-xml-config-file> -r <variable1>=<value1> <variable2>=<value2>

In this case, all elements after the ``-r`` flag are considered variable definitions. Alternatively, the following
syntax can be used. Here all variable names are prefixed with ``--``, and options can be freely mixed:

.. code-block:: bash

    Kassiopeia <path-to-xml-config-file> --<variable1>=<value1> --<variable2>=<value2>

Verbosity levels
----------------

As a quick means to change the output verbosity (i.e. the amount of messages shown on the terminal while the program
is running), the ``-v`` and ``-q`` flags can be used. Each option raises/lowers the verbosity level, so that the
following example would raise the level by one unit. Try it with one of the example XML files below!

.. code-block:: bash

    Kassiopeia <path-to-xml-config-file> -v -q -v

Output files
------------

Upon completion of a simulation, the ROOT output files may be found in the directory
``<kassiopeia-install-path>/output/Kassiopeia`` (where ``<kassiopeia-install-path>`` usually can be replaced by
``$KASPERSYS``.) These output files can then be processed or analyzed by an external program, such as ``root``. If
VTK_ was enabled at build time and the corresponding writer was enabled in the XML file, a set of ``.vtp`` output files
(polygon data in the VTK file format) is created in the same directory.

In order to prematurely terminate a running simulation, *Kassiopeia* provides a user interrupt feature. To terminate the
simulation while allowing *Kassiopeia* to clean up and save the accumulated data to file, the user may type ``Crtl-C``
just once in the terminal. To immediately exit the simulation (without cleaning up or saving data), the user may press
the key combination ``Crtl-C`` again, which leads to program termination.



.. [*] D. Furse *et al.* (2017) New J. Phys. **19** 053012, `doi:10.1088/1367-2630/aa6950 <https://doi.org/10.1088/1367-2630/aa6950>`_
