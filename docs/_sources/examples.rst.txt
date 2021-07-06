Examples and Tests
******************

Running Kassiopeia
------------------

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
""""""""""""""""

As a quick means to change the output verbosity (i.e. the amount of messages shown on the terminal while the program
is running), the ``-v`` and ``-q`` flags can be used. Each option raises/lowers the verbosity level, so that the
following example would raise the level by one unit. Try it with one of the example XML files below!

.. code-block:: bash

    Kassiopeia <path-to-xml-config-file> -v -q -v

Output files
""""""""""""

Upon completion of a simulation, the ROOT output files may be found in the directory
``<kassiopeia-install-path>/output/Kassiopeia`` (where ``<kassiopeia-install-path>`` usually can be replaced by
``$KASPERSYS``.) These output files can then be processed or analyzed by an external program, such as ``root``. If
VTK_ was enabled at build time and the corresponding writer was enabled in the XML file, a set of ``.vtp`` output files
(polygon data in the VTK file format) is created in the same directory.

In order to prematurely terminate a running simulation, *Kassiopeia* provides a user interrupt feature. To terminate the
simulation while allowing *Kassiopeia* to clean up and save the accumulated data to file, the user may type ``Crtl-C``
just once in the terminal. To immediately exit the simulation (without cleaning up or saving data), the user may press
the key combination ``Crtl-C`` again, which leads to program termination.


Example configurations
----------------------

Next in this section, we will investigate a number of pre-configured example files that are shipped with *Kassiopeia*.
These files serve as a working example of various features of the simulation, but also as a reference for your own
configurations. Please take some time to investigate these example files.

The example configurations can be found online at :gh-code:`Kassiopeia/XML/Examples` and are installed to:

    ``$KASPERSYS/config/Kassiopeia/Examples/``

The Dipole Trap Example
"""""""""""""""""""""""

The first example is a simulation of a simple dipole trap, or rather, a magnetic mirror device. It consists of two
dipole magnets positioned some distance apart along the same axis to form a magnetic bottle. To make the simulation
slightly less trivial and more suitable to show *Kassiopeia* features, a ring electrode has been added near the center
of the magnetic bottle emulating a MAC-E filter spectrometer. To run this simulation type:

.. code-block:: bash

    Kassiopeia $KASPERSYS/config/Kassiopeia/Examples/DipoleTrapSimulation.xml

On the first run of this simulation, *Kassiopeia* will call *KEMField* to solve the static electric and magnetic fields
of the geometry and compute zonal harmonic (an axially symmetric approximation) coefficients. Once field solving is
finished, particle tracking of 1 event will commence.

Ignoring the messages from *KEMField*, the terminal output of this example will look something like::

    [KSMAIN NORMAL MESSAGE] ☻  welcome to Kassiopeia 3.7  ☻
    ****[KSRUN NORMAL MESSAGE] processing run 0...
    ********[KSEVENT NORMAL MESSAGE] processing event 0 <generator_uniform>...
    ************[KSTRACK NORMAL MESSAGE] processing track 0 <generator_uniform>...
    [KSNAVIGATOR NORMAL MESSAGE]   child surface <surface_downstream_target> was crossed.
    ************[KSTRACK NORMAL MESSAGE] ...completed track 0 <term_downstream_target> after 48714 steps at <-0.000937309 -0.000478289 0.48>
    ********[KSEVENT NORMAL MESSAGE] ...completed event 0 <generator_uniform>
    ****[KSRUN NORMAL MESSAGE] ...run 0 complete
    [KSMAIN NORMAL MESSAGE] finished!
    [KSMAIN NORMAL MESSAGE] ...finished

Once particle tracking has terminated you will find a ``.root`` output file located at:

    ``$KASPERSYS/output/Kassiopeia/DipoleTrapSimulation.root``

This file contains the data pertaining to the particle's state during tracking, saved in the ROOT TTree format. The
contents of the output file are configured in the XML file. The ROOT file maybe opened for quick visualization and
histogramming using the ROOT TBrowser_ or other suitable tool, or it may be processed by an external analysis program.
As an example, plotting the electric potential experienced by the electron as a function of its $z$ position produces
the following graph.

.. image:: _images/dipole_potential_vs_z.png
   :width: 500pt

For more advanced visualization *Kassiopeia* may be linked against the VTK_ library. If this is done, the
``DipoleTrapSimulation.xml`` example will include a configuration which will open an interactive VTK visualization
window upon completion of the simulation. The output of which shows the electron's track colored by angle between its
momentum vector and the magnetic field. The following image demonstrates the VTK visualization of the simulation.

.. image:: _images/dipole_vtk.png
   :width: 500pt

The Quadrupole Trap Example
"""""""""""""""""""""""""""

The second example to demonstrate the capabilities of *Kassiopeia* is that of a quadrupole (Penning) trap. This sort of
trap is similar to those which are used to measure the electron $g$-factor to extreme precision. To run this example,
locate the XML file in the config directory, and at the command prompt enter:

.. code-block:: bash

    Kassiopeia $KASPERSYS/config/Kassiopeia/Examples/QuadrupoleTrapSimulation.xml

This example also demonstrates the incorporation of discrete interactions, such as scattering off residual gas. If VTK_
is used, upon the completion of the simulation a visualization window will appear. An example of this shown in the
following figure. The large green tube is the solenoid magnet, while the amber hyperboloid surfaces within it are the
electrode surfaces. The electron tracks can be seen as short lines at the center.

.. image:: _images/quadrupole_vtk.png
   :width: 500pt

Furthermore, a very simple analysis program example ``QuadrupoleTrapAnalysis`` can be run on the resulting ``.root``
file. To do this, execute the following after the output file was created:

.. code-block:: bash

    QuadrupoleTrapAnalysis $KASPERSYS/output/Kassiopeia/QuadrupoleTrapSimulation.root

The output of which should be something to the effect of::

    extrema for track <1.43523>

This program can be used as a basis for more advanced analysis programs, as it demonstrates the methods needed to
iterate over the particle tracking data stored in a ROOT TTree file. It is also possible to access the ROOT TTree data
by other means, e.g. using Python scripts and the PyROOT_ or uproot_ modules, but this is out of scope for this section.

The Photomultiplier Tube Example
""""""""""""""""""""""""""""""""

As a demonstration of some of the more advanced features of *Kassiopeia* (particularly its 3D capabilities), an example
of particle tracking in a photomultiplier tube is also included. This convifuration was also featured in the
*Kassiopeia* paper [*]_.

Since the dimensions of the linear system that needs to be solved in order to compute the electric field is rather large
(~150K mesh elements), the initialization of the electric field may take some time. If the user has the appropriate
device (e.g. a GPU) it is recommended that the field solving sub-module *KEMField* is augmented with OpenCL in order to
take advantage of hardware acceleration. This is done by setting the ``KEMField_USE_OpenCL`` flag in the build stage.

To run this simulation, type:

.. code-block:: bash

    Kassiopeia $KASPERSYS/config/Kassiopeia/Examples/PhotoMultiplierTubeSimulation.xml

Depending on the capability of your computer this example may take several hours to run, and you may want to execute it
overnight. If you have enabled VTK_, an ``.vtp`` output file called:

    ``$KASPERSYS/output/Kassiopeia/PhotoMultiplierTubeSimulation.vtpStep.vtp``

will be created. This file stores the particle step data for visualization using the VTK polydata format. Additionally,
a file called ``PhotomultiplierTube.vtp`` will be created in the directory from which *Kassiopeia* was called. This file
stores visualization data about the electrode mesh elements used by *KEMField*. Both of these files can be opened in the
external program Paraview_ for data selection and viewing, or other suitable software. An example is shown in the
following figure.

.. image:: _images/pmt_paraview.png
   :width: 500pt


The Mesh Simulation Example
"""""""""""""""""""""""""""

The mesh simulation uses a geometry from an external STL_ file, which is a format widely used in 3D design software.
The external geometry must provide a surface mesh in order to be usable with *KEMField* and *Kassiopeia*. In this
example, an electric field is defined by two copies of the so-called `Menger sponge` cubes that are placed next to each
other. Particles are tracked along a linear trajectory, which are reflected when they hit one of the cube surfaces.

.. image:: _images/mesh_simulation.png
   :width: 500pt

Other Examples
""""""""""""""

Some other examples which explore other concepts
also distributed with Kassiopeia, and are described in the following table.


.. |ana| image:: _images/analytic_trap.png
   :scale: 30%
   :align: middle

.. |toric| image:: _images/toric.png
   :scale: 24%
   :align: middle

.. |dmvtk| image:: _images/dipole_meshed_vtk.png
   :scale: 30%
   :align: middle


+---------------------------------------------------------------------------------------------------------+
| Other simulation examples                                                                               |
+-----------------------------------------+---------------------------------------------------------------+
| File                                    |  Description                                                  |
+=========================================+===============================================================+
| ``AnalyticSimulation.xml``              | Quadrupole ion/electron trap (similar to the original         |
|                                         | ``QuadrupoleTrapSimulation.xml``. However, the magnetic       |
|  |ana|                                  | field is completely uniform and the and the electric          |
|                                         | field is described analytically as an ideal quadrupole.       |
+-----------------------------------------+---------------------------------------------------------------+
| ``ToricTrapSimulation.xml``             | This is a simulation of an electron trapped in a magnetic     |
|                                         | torus (similar to a Tokamak reactor), and it demonstrates the |
|  |toric|                                | identification of surface intersections, as well as particle  |
|                                         | drift in non-homogeneous fields.                              |
+-----------------------------------------+---------------------------------------------------------------+
| ``DipoleTrapMeshedSpaceSimulation.xml`` | This simulation has the same fields as the original           |
|                                         | ``DipoleTrapSimulation.xml`. However, there are additional    |
|  |dmvtk|                                | (meshed, but non-interacting) surfaces present to demonstrate |
|                                         | navigation in a complicated geometry using the meshed-surface |
|                                         | octree-based navigator.                                       |
+-----------------------------------------+---------------------------------------------------------------+


.. _VTK: http://www.vtk.org/
.. _Paraview: http://www.paraview.org/
.. _TBrowser: https://root.cern.ch/doc/master/classTBrowser.html
.. _PyROOT: https://root.cern/manual/python/
.. _uproot: https://pypi.org/project/uproot/
.. _STL: https://en.wikipedia.org/wiki/STL_%28file_format%29

.. [*] D. Furse *et al.* (2017) New J. Phys. **19** 053012, `doi:10.1088/1367-2630/aa6950 <https://doi.org/10.1088/1367-2630/aa6950>`_
