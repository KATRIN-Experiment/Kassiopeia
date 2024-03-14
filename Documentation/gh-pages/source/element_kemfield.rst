.. _KEMField:

KEMField (fields) - <kemfield>
===========================



**KEMField** is a toolkit that allows users to solve electrostatic and magnetostatic fields from user-defined inputs. 
Detailed explanations on installation and implementation are available in the manual displayed at the bottom of this chapter. 
Alternatively the manual is available for download :download:`here <../../../KEMField/Documentation/manual/manual.pdf>` 
and can also be found in the `Github repository <https://github.com/KATRIN-Experiment/Kassiopeia/>`_. 


The field elements all live within the *KEMField* element and must be placed with start and end tags of the form:

.. code-block:: xml

    <kemfield>
        <!-- complete description of the kemfield configuration element here -->
    </kemfield>

Note that in some configuration files, you may find the "legacy style" setup where the field elements are defined under
the *Kassiopeia* element (see below). Although both variants are supported, it is recommended to follow the one
described here.

Once the simulation geometry has been specified, the user may describe the types of electric and magnetic fields they
wish associate with each geometric object. The field package *KEMField* takes care of solving the boundary value problem
and computing the fields for electrostatic problems. It also handles the magnetic field computation from static current
distributions.

Fast field calculation methods are available for axially symmetric (zonal harmonics) and three dimensional problems
(fast multipole method). The abstract base classes responsible for electric and magnetic fields in *Kassiopeia* are
**KSElectricField** and **KSMagneticField** respectively, which interface with the corresponding
implementations in *KEMField*.

For example, in the ``DipoleTrapSimulation.xml`` example the electric and magnetic fields are axially symmetric and can
be computed using the zonal harmonics expansion.

Electric fields
-----------

To specify the electric field, the geometric surfaces which are electrically active must be listed in the ``surfaces``
element. It is important that the surfaces which are specified have a mesh extension and a boundary type extension. If
either of these extensions are missing from the specified surface, they will not be included in the electrostatics
problem. A boundary element mesh is needed to solve the Laplace equation using the boundary element method. Each element
of the mesh inherits its parent surface's boundary condition type.

Both a method to solve the Laplace boundary value problem (a ``bem_solver``), and a method by which to compute the
fields from the resulting charge densities must be given (a ``field_sovler``). In the following example we use a
``robin_hood_bem_solver`` and a ``zonal_harmonic_field_solver``:

.. code-block:: xml

    <electrostatic_field
            name="field_electrostatic"
            directory="[KEMFIELD_CACHE]"
            file="DipoleTrapElectrodes.kbd"
            system="world/dipole_trap"
            surfaces="world/dipole_trap/@electrode_tag"
            symmetry="axial"
            >
        <robin_hood_bem_solver
                integrator="analytic"
                tolerance="1.e-10"
                check_sub_interval="100"
                display_interval="1"
                cache_matrix_elements="true"
        />
        <zonal_harmonic_field_solver
                number_of_bifurcations="-1"
                convergence_ratio=".99"
                convergence_parameter="1.e-15"
                proximity_to_sourcepoint="1.e-12"
                number_of_central_coefficients="500"
                use_fractional_central_sourcepoint_spacing="false"
                central_sourcepoint_spacing="1.e-3"
                central_sourcepoint_start="-5.2e-1"
                central_sourcepoint_end="5.2e-1"
                number_of_remote_coefficients="200"
                remote_sourcepoint_start="-5.e-2"
                remote_sourcepoint_end="5.e-2"
        />
    </electrostatic_field>

It is also important that geometric elements be meshed appropriately with respect to symmetry. In the case that the user
wishes to use zonal harmonic field calculation routines, an ``axial_mesh`` must be used. If a normal (3D) mesh is used,
zonal harmonics cannot function. Different mesh/symmetry types cannot be combined within the same electric field solving
element. The symmetry of the electric field model is set by the ``symmetry`` attribute.

The zonal-harmonic solver offers many parameters to fine-tune the applied approximation. The example above lists mostly
default values. The most important parameter is probably the distance of the "source points", which provide the basis
for the approximation. The example above defines a spacing of 1 mm along the z-axis.

In the three-dimensional mesh case, either an integrating field solver, or a fast multipole field solver may be used.
The integrating field solver may be specified through inclusion of the element:

.. code-block:: xml

    <integrating_field_solver/>

within the the ``electrostatic_field`` element (replacing the ``zonal_harmonic_field_solver`` in the example above).
As the integrating field solver is quite simple, it does not require additional parameters.

The fast multipole field solver on the other hand is somewhat more complex and requires a relatively large set of
additional parameters to be specified in order to configure its use according to the user's desired level of accuracy
and computational effort.

For a complete list and description of the XML bindings available for the electric field solving routines, navigate to
the directory ``$KASPERSYS/config/KEMField/Complete``. The file ``ElectricFields.xml`` has examples of the binding for
initializing electric field problems (see :gh-code:`KEMField/Source/XML/Complete/ElectricFields.xml`.)

Magnetic fields
----------

The specification of the magnetic field solving routines is considerably simpler since there is no need to solve a
boundary value problem before hand. There are essentially two choices for solving magnetic fields from static current
distributions: The zonal harmonics method for use with axially symmetric current sources, and the integrating magnetic
field solver which can be used on geometries with more arbitrary distributions of current. Unlike electric fields,
magnetic fields can contain components with both axially symmetric and non-axially symmetric elements within the same
region with no adverse effects.

The following example uses the zonal harmonics method to compute the magnetic field:

.. code-block:: xml

    <electromagnet_field
            name="field_electromagnet"
            directory="[KEMFIELD_CACHE]"
            file="DipoleTrapMagnets.kbd"
            system="world/dipole_trap"
            spaces="world/dipole_trap/@magnet_tag"
            >
        <zonal_harmonic_field_solver
                number_of_bifurcations="-1"
                convergence_ratio=".99"
                convergence_parameter="1.e-15"
                proximity_to_sourcepoint="1.e-12"
                number_of_central_coefficients="500"
                use_fractional_central_sourcepoint_spacing="true"
                central_sourcepoint_fractional_distance="1e-2"
                central_sourcepoint_spacing="1.e-3"
                number_of_remote_coefficients="200"
                remote_sourcepoint_start="-5.e-2"
                remote_sourcepoint_end="5.e-2"
        />
    </electromagnet_field>

Note that although the zonal harmonics solver allows a faster calculation of the electromagnetic fields, but requires
some initialization time to compute the source points. Depending on the simulation, the overall computation time could
be lower when using the integrating solver instead.

Also, please note that only three *KGeoBag* shapes can be used to create electromagnets: cylinder surface, cylinder tube
space, and rod space. For details, see the above section `Extensions`. If other shapes are added to the electromagnet
field elemenet, they will not be recognized as magnet geometries. When using rod spaces, the resulting magnet element
will be a "line current" that does not allow any zonal harmonic approximation and is always solved directly.

A complete list and set of examples of the XML bindings for magnetic fields can be found in the file
``$KASPERSYS/config/KEMField/Complete/MagneticFields.xml`` (see :gh-code:`KEMField/Source/XML/Complete/MagneticFields.xml`.)

Further documentation on the exact methods and parameters used in *KEMField* can be found in [2] and [3].

.. _kemfield-visualization:

Visualization
---------------

The *KEMField* modules provides a special visualization that is only available for electrostatic geometries. In contrast
to the geometry viewers from *KGeoBag*, the *KEMField* viewer also includes extra information about the mesh elements,
the applied electric potentials, and the calculated charge densities. It is therefore extremely valuable for the design
of such geometries.

Electrode Geometry
~~~~~~~~~~~~~~~~~~~~~

The viewer is instantiated with the XML element ``<viewer>`` under the ``<ksfield_electrostatic>`` or ``<electrostatic_field>``
tag. For example, expanding the ``DipoleTrapSimulation.xml`` file:

.. code-block:: xml

    <kemfield>
        <electrostatic_field
                name="field_electrostatic"
                file="DipoleTrapElectrodes.kbd"
                system="world/dipole_trap"
                surfaces="world/dipole_trap/@electrode_tag"
                symmetry="axial"
            >
            <robin_hood_bem_solver
                 integrator="analytic"
                 tolerance="1.e-10"
                 check_sub_interval="100"
                 display_interval="1"
                 cache_matrix_elements="true"
            />
            <viewer
                 file="DipoleTrapElectrodes.vtp"
                 save="true"
                 view="true"
                 preprocessing="false"
                 postprocessing="true"
            />
        </electrostatic_field>
    </kemfield>

The options ``save`` and ``view`` specify if an output file should be written to the given filename, and if a viewer
window should be shown. The options ``preprocessing`` and ``postprocessing`` indicate if the visualization is to be
performed before or after calculating the charge densities (if both are true, the visualization is performed twice).

Field maps
~~~~~~~~~~

Although not primarily a visualization feature, the option to compute electric and magnetic field maps with *KEMField*
can also be used to provide input for the ParaView_ software that can be combined with other visualization output files.
Field maps can be calculated in 2D or 3D mode, and both variants can readily be used in ParaView.

The example below will generate a 2D map of the magnetic and electric field:

.. code-block:: xml

    <kemfield>
        <magnetic_fieldmap_calculator
            name="b_fieldmap_calculator"
            field="field_electromagnet"
            file="DipoleTrapMagnetic.vti"
            directory="[output_path]"
            force_update="false"
            compute_gradient="false"
            center="0 0 0"
            length="5e-1 0 1.0"
            spacing="0.01 0.01 0.01"
            mirror_x="true"
            mirror_y="true"
            mirror_z="false"
        />

        <electric_potentialmap_calculator
            name="e_fieldmap_calculator"
            field="field_electrostatic"
            file="DipoleTrapElectric-XZ.vti"
            directory="[output_path]"
            force_update="false"
            compute_field="true"
            center="0 0 0"
            length="5e-1 0.0 1.0"
            spacing="0.01 0.01 0.01"
            mirror_x="true"
            mirror_y="true"
            mirror_z="false"
        />
    </kemfield>

The output files will only be generated once and the computation is skipped if a file under the same name exists. To
force an update, either delete the file or set ``force_update`` to true. The parameters ``center``, ``length`` and
``spacing`` define the bounds and dimensions of the map. (In this example, a 2D map will be created because one of
the dimensions is equal to zero.) To speed up the computation, it is possible to exclude the magnetic field gradient
(``compute_gradient``) or electric field (``compute_field``), or to make use of existing symmetries in either dimension.
Note that the symmetry is not checked against the actual geometry, so it's a responsibility of the user to set this up
correctly.


.. pdf-include:: ./PDFs/manual.pdf
    :width: 100%
    :height: 800px



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
.. _PyVista: https://www.pyvista.org/


.. rubric:: Footnotes

[1] Daniel Lawrence Furse. Techniques for direct neutrino mass measurement utilizing tritium [beta]-decay. PhD thesis, Massachusetts Institute of Technology, 2015.

[2] Thomas Corona. Methodology and application of high performance electrostatic field simulation in the KATRIN experiment. PhD thesis, University of North Carolina, Chapel Hill, 2014.

[3] John P. Barrett. A Spatially Resolved Study of the KATRIN Main Spectrometer Using a Novel Fast Multipole Method. PhD thesis, Massachusetts Institute of Technology, 2016.
