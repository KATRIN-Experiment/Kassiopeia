Examples and Tests
******************

Running Kassiopeia
------------------

After installation, and assuming the proper environmental variables have been set by running
the script ``kasperenv.sh``, *Kassiopeia* can be run directly from the command prompt. To run
a *Kassiopeia* simulation, the useage is simply::

    Kassiopeia <path-to-xml-config-file>

*Kassiopeia* also includes advanced variable replacement functionality, which makes it easy to
modify one or many simulation parameters on-the-fly. This can be done with the following useage::

    Kassiopeia <path-to-xml-config-file> -r <variable1>=<value1> <variable2>=<value2>

Upon completion of a simulation, the ROOT output files
may be found in the directory ``<kassiopeia-install-path>/output/Kassiopeia``. These output
file can then be processed or analyzed by an external program. In order to prematurely terminate
a running simulation *Kassiopeia* provides a user interrupt feature. To terminate the simulation
early, but allow *Kassiopeia* to clean up and save the accumulated data to file, the user
may type ``crtl-C`` just once in the terminal. To immediately exit the simulation (without cleaning
up or saving data) the user may type ``crtl-C`` twice.

The Dipole Trap Example
-----------------------

The first example is the simplest is a simulation of a dipole trap, or rather, a magnetic mirror device.
It consists of a two dipole magnets positioned some distance apart
along the same axis to form a magnetic bottle. To make the simulation slightly less trivial but still
simple, a ring electrode has been added near the center of the magnetic bottle
emulating a MAC-E filter spectrometer. To run this simulation, from the $KASPERSYS directory type::

    Kassiopeia ./config/Kassiopeia/Examples/DipoleTrapSimulation.xml

On the first run of this simulation, *Kassiopeia* will call *KEMField* to solve the
static electric and magnetic fields of the geometry and compute a zonal
harmonic (axially symmetric approximation) field map. Once field solving
is finished, particle tracking of 1 event will commence.

Ignoring the messages from *KEMField*, the terminal output of this example will look
something like::

    [KSMAIN NORMAL MESSAGE] ☻  welcome to Kassiopeia 3.3  ☻
    ****[KSRUN NORMAL MESSAGE] processing run 0...
    ********[KSEVENT NORMAL MESSAGE] processing event 0 <generator_uniform>...
    ************[KSTRACK NORMAL MESSAGE] processing track 0 <generator_uniform>...
    [KSNAVIGATOR NORMAL MESSAGE]   child surface <surface_downstream_target> was crossed.00106397, k = 5.44591, e = 7.08033))
    ************[KSTRACK NORMAL MESSAGE] ...completed track 0 <term_downstream_target> after 48714 steps at <-0.000937309 -0.000478289 0.48>
    ********[KSEVENT NORMAL MESSAGE] ...completed event 0 <generator_uniform>
    ****[KSRUN NORMAL MESSAGE] ...run 0 complete
    [KSMAIN NORMAL MESSAGE] finished!
    [KSMAIN NORMAL MESSAGE] ...finished

Once particle tracking has terminated you will find a .root output file located
at::

    <kassiopeia-install-path>/output/Kassiopeia/DipoleTrapSimulation.root

This file contains all the data (saved as in a ROOT TTree format)
pertaining to the particle's state during tracking. This file maybe opened
for quick visualization and histogramming using a ROOT TBrowser, or it may
be processes by an external analysis program. As an example, plotting the
electric potential experienced by the electron as a function of its $z$
position produces the following graph.

.. image:: dipole_potential_vs_z.png
   :width: 500pt


For more advanced visualization *Kassiopeia* may be linked against the VTK_ library.
This can be done by toggling the the cmake variable ``Kassiopeia_USE_VTK``. If this is done,
the DipoleTrapSimulation.xml example will include a configuration which
will open an interactive VTK visualization window upon completion of the simulation. The output of which
shows the electron's track colored by angle between its momentum vector and the magnetic field.
The following image demonstrates the VTK visualization of the Dipole trap simulation.

.. image:: dipole_vtk.png
   :width: 500pt

The Quadrupole Trap Example
---------------------------

The second example to demonstrate the capabilities of *Kassiopeia* is that of a quadrupole (Penning) trap.
This sort of trap is similar to those which are used to measure the electron $g$-factor to extreme precision.
To run this example first navigate to the installation directory using ``cd $KASPERSYS``,
and then at the command prompt enter::

    Kassiopeia ./config/Kassiopeia/Examples/QuadrupoleTrapSimulation.xml

This example also demonstrates the incorporation of discrete interactions, such as scattering
off residual gas. If VTK is used, upon the completion of particle track a visualization window will appear. An
example of this shown in the following figure. The large green tube is the solenoid magnet, while the amber hyperboloid surfaces within it are
the electrode surfaces. The electron tracks can be seen as short lines at the center.

.. image:: quadrupole_vtk.png
   :width: 500pt

Furthermore, a very simple analysis program example ``QuadrupoleTrapAnalysis`` can be run on the resulting .root
file, by executing::

    QuadrupoleTrapAnalysis ./output/Kassiopeia/QuadrupoleTrapSimulation.root

The output of which should be something to the effect of::

    extrema for track <1.43523>

This program can be used as a basis for more advanced analysis programs, as it demonstrates the methods
needed to iterate over the particle tracking data stored in a ROOT TTree file.

The Photomultiplier Tube Example
--------------------------------

As a demonstration of some of the more advanced features of *Kassiopeia* (particularly its 3D capabilities)
an example of particle tracking in a photomultiplier tube is also included. Since the dimensions of the linear
system that needs to be solved in order to compute the electric field is rather large (~150K mesh elements),
the initialization of the electric field may take some time. If the user has the appropriate device (e.g. a GPU)
it is recommended that the field solving sub-module *KEMField* be augmented with OpenCL in order
to take advantage of hardware based acceleration. This can be done by setting the KEMField_USE_OPENCL
flag to ON during cmake configuration.

To run this simulation, from the $KASPERSYS directory type::

    Kassiopeia ./config/Kassiopeia/Examples/PhotoMultiplierTubeSimulation.xml

Depending on the capability of your computer this example may take several hours to run, you may want to
execute it overnight. If you have enabled VTK_, an .vtp output file called::

    <kassiopeia-install-path>/output/Kassiopeia/PhotoMultiplierTubeSimulation.vtpStep.vtp

will be created. This file stores the particle step data for visualization using the VTK polydata format.
Additionally, a file called ``PhotomultiplierTube.vtp`` will be created in the directory from which *Kassiopeia*
was called. This file stores visualization data about the electrode BEM mesh elements. Both of these
files can be opened in the the external program Paraview_ for data selection and viewing. An
example is shown in the following figure.

.. image:: pmt_paraview.png
   :width: 500pt

Other Examples
--------------

Some other examples which explore other concepts
also distributed with Kassiopeia, and are described in the following table.


.. |dmvtk| image:: dipole_meshed_vtk.png
   :scale: 30%
   :align: middle

.. |ana| image:: analytic_trap.png
   :scale: 30%
   :align: middle

.. |toric| image:: toric.png
   :scale: 24%
   :align: middle


+-----------------------------------------------------------------------------------------------------+
| Other simulation examples                                                                           |
+-------------------------------------+---------------------------------------------------------------+
| File                                |  Description                                                  |
+=====================================+===============================================================+
| AnalyticSimulation.xml              | Quadrupole ion/electron trap (similar to the original         |
|                                     | QuadrupoleTrapSimulation.xml. However, the magnetic           |
|  |ana|                              | field is completely uniform and the and the electric          |
|                                     | field is described analytically as an ideal quadrupole.       |
+-------------------------------------+---------------------------------------------------------------+
| ToricTrapSimulation.xml             | This is a simulation of an electron trapped in a magnetic     |
|                                     | torus (similar to a tokamak), and demonstrates the            |
| |toric|                             | identification of surface intersections, as well as particle  |
|                                     | drift in non-homogeneous fields.                              |
+-------------------------------------+---------------------------------------------------------------+
| DipoleTrapMeshedSpaceSimulation.xml | This simulation has the same fields as the original           |
|                                     | dipole trap simulation. However, there are many additional    |
|  |dmvtk|                            | (meshed, but non-interacting) surfaces present to demonstrate |
|                                     | navigation in a complicated geometry using the meshed-surface |
|                                     | octree-based navigator.                                       |
+-------------------------------------+---------------------------------------------------------------+



.. _VTK: http://www.vtk.org/
.. _Paraview: http://www.paraview.org
