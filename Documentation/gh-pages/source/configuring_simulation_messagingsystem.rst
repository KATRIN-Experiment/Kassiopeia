
Messaging System
=================

*Kassiopeia* provides a very granular means of reporting and logging simulation details of interest. This feature is
particularly useful when modifying the code and debugging specific features. For example, at the top of the file
``QuadrupoleTrapSimulation.xml`` you can find section describing the verbosity of each simulation element and the
location of the logging file (as defined by the variable ``log_path`` and the ``<file>`` element):

.. code-block:: xml

    <define name="log_path" value="[KASPERSYS]/log/Kassiopeia"/>

    <messages>

        <file path="[log_path]" base="QuadrupoleTrapLog.txt"/>

        <message key="k_file" terminal="normal" log="warning"/>
        <message key="k_initialization" terminal="normal" log="warning"/>

        <message key="kg_core" terminal="normal" log="warning"/>
        <message key="kg_shape" terminal="normal" log="warning"/>
        <message key="kg_mesh" terminal="normal" log="warning"/>
        <message key="kg_axial_mesh" terminal="normal" log="warning"/>

        <message key="ks_object" terminal="debug" log="normal"/>
        <message key="ks_operator" terminal="debug" log="normal"/>
        <message key="ks_field" terminal="debug" log="normal"/>
        <message key="ks_geometry" terminal="debug" log="normal"/>
        <message key="ks_generator" terminal="debug" log="normal"/>
        <message key="ks_trajectory" terminal="debug" log="normal"/>
        <message key="ks_interaction" terminal="debug" log="normal"/>
        <message key="ks_navigator" terminal="debug" log="normal"/>
        <message key="ks_terminator" terminal="debug" log="normal"/>
        <message key="ks_writer" terminal="debug" log="normal"/>
        <message key="ks_main" terminal="debug" log="normal"/>
        <message key="ks_run" terminal="debug" log="normal"/>
        <message key="ks_event" terminal="debug" log="normal"/>
        <message key="ks_track" terminal="debug" log="normal"/>
        <message key="ks_step" terminal="debug" log="normal"/>

    </messages>

For the verbosity settings, you can independently set the verbosity that you see in the terminal and the verbosity that
is put into log files. Furthermore, you can do that for each different part of *Kassiopeia* and the other modules. For
example, if you want a lot of detail on what's happening in the navigation routines, you can increase the verbosity for
only that part of *Kassiopeia*, without being flooded with messages from everything else. The different sources are
define by the ``key`` attribute of the ``<message>`` element, and explained in the table below.

+--------------------------------------------------------------------------------------------------------------------------+
| Message sources                                                                                                          |
+-----------------------+-------------+-----------------------------------+------------------------------------------------+
| Key                   | Module      | Location                          | Description                                    |
+=======================+=============+===================================+================================================+
| ``k_file``            | Kommon      | File                              | File handling                                  |
+-----------------------+-------------+-----------------------------------+------------------------------------------------+
| ``k_initialization``  | Kommon      | Initialization                    | XML initialization and processing              |
+-----------------------+-------------+-----------------------------------+------------------------------------------------+
| ``k_utility``         | Kommon      | Utility                           | Utility functions                              |
+-----------------------+-------------+-----------------------------------+------------------------------------------------+
| ``kem_bindings``      | KEMField    | Bindings                          | XML bindings                                   |
+-----------------------+-------------+-----------------------------------+------------------------------------------------+
| ``kem_core``          | KEMField    | Core                              | Core functionality                             |
+-----------------------+-------------+-----------------------------------+------------------------------------------------+
| ``kg_bindings``       | KGeoBag     | Bindings                          | XML bindings                                   |
+-----------------------+-------------+-----------------------------------+------------------------------------------------+
| ``kg_core``           | KGeoBag     | Core                              | Core functionality                             |
+-----------------------+-------------+-----------------------------------+------------------------------------------------+
| ``kg_axial_mesh``     | KGeoBag     | Extensions/AxialMesh              | Axially symmetric meshing                      |
+-----------------------+-------------+-----------------------------------+------------------------------------------------+
| ``kg_drmesh``         | KGeoBag     | Extensions/DiscreteRotationalMesh | Rotationally discrete meshing                  |
+-----------------------+-------------+-----------------------------------+------------------------------------------------+
| ``kg_mesh``           | KGeoBag     | Extensions/Mesh                   | Asymmetric meshing                             |
+-----------------------+-------------+-----------------------------------+------------------------------------------------+
| ``kg_metrics``        | KGeoBag     | Extensions/Metrics                | Metrics calculation (volumes & areas)          |
+-----------------------+-------------+-----------------------------------+------------------------------------------------+
| ``kg_random``         | KGeoBag     | Extensions/Random                 | Random generator functions                     |
+-----------------------+-------------+-----------------------------------+------------------------------------------------+
| ``kg_math``           | KGeoBag     | Math                              | Mathematical functions                         |
+-----------------------+-------------+-----------------------------------+------------------------------------------------+
| ``kg_shape``          | KGeoBag     | Shapes                            | Geometric shapes                               |
+-----------------------+-------------+-----------------------------------+------------------------------------------------+
| ``kg_vis``            | KGeoBag     | Visualization                     | Visualization (VTK_, ROOT_)                    |
+-----------------------+-------------+-----------------------------------+------------------------------------------------+
| ``ks_bindings``       | Kassiopeia  | Bindings                          | XML bindings                                   |
+-----------------------+-------------+-----------------------------------+------------------------------------------------+
| ``ks_field``          | Kassiopeia  | Fields                            | Field calculation                              |
+-----------------------+-------------+-----------------------------------+------------------------------------------------+
| ``ks_generator``      | Kassiopeia  | Generators                        | Particle generation                            |
+-----------------------+-------------+-----------------------------------+------------------------------------------------+
| ``ks_geometry``       | Kassiopeia  | Geometry                          | Geometry handling                              |
+-----------------------+-------------+-----------------------------------+------------------------------------------------+
| ``ks_interaction``    | Kassiopeia  | Interactions                      | Particle interactions                          |
+-----------------------+-------------+-----------------------------------+------------------------------------------------+
| ``ks_math``           | Kassiopeia  | Math                              | Mathematical functions                         |
+-----------------------+-------------+-----------------------------------+------------------------------------------------+
| ``ks_modifier``       | Kassiopeia  | Modifiers                         | Trajectory modifiers                           |
+-----------------------+-------------+-----------------------------------+------------------------------------------------+
| ``ks_navigator``      | Kassiopeia  | Navigators                        | Particle navigation                            |
+-----------------------+-------------+-----------------------------------+------------------------------------------------+
| ``ks_object``         | Kassiopeia  | Objects                           | Dynamic command interface                      |
+-----------------------+-------------+-----------------------------------+------------------------------------------------+
| ``ks_operator``       | Kassiopeia  | Operators                         | Core functionality, particle state             |
+-----------------------+-------------+-----------------------------------+------------------------------------------------+
| ``ks_reader``         | Kassiopeia  | Readers                           | File reading                                   |
+-----------------------+-------------+-----------------------------------+------------------------------------------------+
| ``ks_main``           | Kassiopeia  | Simulation                        | Simulation execution                           |
+-----------------------+-------------+-----------------------------------+------------------------------------------------+
| ``ks_run``            | Kassiopeia  | Simulation                        | Simulation progress, "run" level               |
+-----------------------+-------------+-----------------------------------+------------------------------------------------+
| ``ks_event``          | Kassiopeia  | Simulation                        | Simulation progress, "event" level             |
+-----------------------+-------------+-----------------------------------+------------------------------------------------+
| ``ks_track``          | Kassiopeia  | Simulation                        | Simulation progress, "track" level             |
+-----------------------+-------------+-----------------------------------+------------------------------------------------+
| ``ks_step``           | Kassiopeia  | Simulation                        | Simulation progress, "step" level              |
+-----------------------+-------------+-----------------------------------+------------------------------------------------+
| ``ks_terminator``     | Kassiopeia  | Terminators                       | Particle termination                           |
+-----------------------+-------------+-----------------------------------+------------------------------------------------+
| ``ks_trajectory``     | Kassiopeia  | Trajectories                      | Trajectory calculation                         |
+-----------------------+-------------+-----------------------------------+------------------------------------------------+
| ``ks_utility``        | Kassiopeia  | Utility                           | Utility functions                              |
+-----------------------+-------------+-----------------------------------+------------------------------------------------+
| ``ks_vis``            | Kassiopeia  | Visualization                     | Visualization (VTK_, ROOT_)                    |
+-----------------------+-------------+-----------------------------------+------------------------------------------------+
| ``ks_writer``         | Kassiopeia  | Writers                           | File writing                                   |
+-----------------------+-------------+-----------------------------------+------------------------------------------------+

The different parts of the code are explained further below, along with XML configuration examples.

Verbosity levels
~~~~~~~~~~~~~~~~

There are five possible verbosity levels, they are ``debug``, ``info``, ``normal``, ``warning`` and ``error``. Of these,
``error`` is the least verbose, only reporting on fatal errors that terminate the simulation. The ``normal`` mode will
include a relatively small set of details in addition to any warnings (this is the default), while ``debug`` will
provide an extremely extensive description of the state of the simulation as it progresses.

Note that the ``debug`` setting is a special case: Since there is so much additional information provided by this
setting, it substantially slows down the speed of the simulation even when the messages are not printed or saved to the
log file. In order to avoid unnecessarily slowing down *Kassiopeia*, the debug output is completely disabled unless it
is explicitly enabled in the build by enabling the CMake option ``Kassiopeia_ENABLE_DEBUG`` during configuration (and
the corresponding options for other modules.)

As mentioned earlier, the verbosity level can also be changed by the command line arguments ``-v`` and ``-q``, which
raise or lower the verbosity level. However, this only works for sources that have not been configured explicitely
in the ``<messages>`` section.

Additional logging
~~~~~~~~~~~~~~~~~~

The description above applies to the *KMessage* interface, which is configured through XML files. In addition, some code
uses the independent *KLogger* interface. If *Kassiopeia* was compiled with Log4CXX_ enabled at build time, the KLogger
interface can be configured through its own configuration file, which is located at:

    ``$KASPERSYS/config/Kommon/log4cxx.properties``

It allows flexible logging configuration of different parts of the code, including changing the verbosity level,
redirecting output to a log file, or customizing the message format.

.. note::

    In *Kassiopeia*, *KEMField* and *KGeoBag*, most messages use the *KMessage* interface.


.. _TFormula: http://root.cern.ch/root/htmldoc/TFormula.html
.. _TMath: http://root.cern.ch/root/htmldoc/TMath.html
.. _PDG: http://pdg.lbl.gov/mc_particle_id_contents.html
.. _Paraview: http://www.paraview.org/
.. _ROOT: https://root.cern.ch/
.. _VTK: http://www.vtk.org/
.. _MKS: https://scienceworld.wolfram.com/physics/MKS.html
.. _XML: https://www.w3.org/TR/xml11/
.. _Xpath: https://www.w3.org/TR/xpath-10/
.. _TinyExpr: https://github.com/codeplea/tinyexpr/
.. _Log4CXX: https://logging.apache.org/log4cxx/
