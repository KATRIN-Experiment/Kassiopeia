
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
