.. _visualization-label:

Visualization Techniques
************************

The *Kasper* software provides some visualization techniques that can help with the design and optimization of a
simulation, and with interpreting its results. In general, there are options for 2D visualization using the ROOT_
software and for 3D visualization using the VTK_ toolkit. The VTK output files can also be viewed and combined with
external software such as ParaView_.

The examples in this section are based on the ``DipoleTrapSimulation.xml`` file, which may be extended accordingly to
test the features explained here.


KGeoBag visualization
---------------------

The *KGeoBag* module provides a set of stand-alone visualization tools that are described under :ref:`tools`. These
are suited to display the simulation geometry and other geometric elements, such as the mesh used for field calculation
and/or navigation.

In addition, the geometry visualization can also be defined in the configuration file. In this case, output files
may be produced before or after performing the simulation, and a visualization window can be shown as well. Note that
the visualization window blocks the application until it is closed, so it is not advised to use this feature in a
scripted environment. The commandline option ``-b`` (or ``-batch``) will prevent any visualization windows to appear
regardless of the setting in the configuration file, e.g.:

.. code-block:: bash

    Kassiopeia -b DipoleTrapSimulation.xml

The *KGeoBag* module provides painter classes for the geometry which are covered below in the *Kassiopeia* section.
In addition, the mesh geometry can be viewed as well:

.. code-block:: xml

    <vtk_window
            name="vtk_window"
            enable_display="true"
            enable_write="true"
            frame_title="KGeoBag Visualization"
        >
        <vtk_axial_mesh_painter
            name="vtk_axial_mesh_painter"
            surfaces="world/dipole_trap/@electrode_tag"
            color_mode="area"
        />
    </vtk_window>

The axial mesh painter needs a defined mesh (``<axial_mesh>`` XML element, see :ref:`configuration`). An
``<vtk_mesh_painter>`` exists as well, to be used with an asymmetric mesh (defined via ``<mesh>``.)

Python
~~~~~~

It is possible to draw a geometry visualization in Python. This is especially useful if you run your analysis in Python as well (see :ref:`output-label` for examples.)

The PyVista_ Python package makes it easy to operate on the VTK_ output files that are produced by *KGeoBag* and the other *Kasper* modules. (Note that this method only works if VTK_ was enabled at build time.) In the XML snippets above, the different VTK painters will produce output files if the ``enable_write`` attribute is set. These files contain the 3D geometry in the `vtp` format and can be read in Python. In order to get a 2D view of the geometry, one also needs to create a slice that transforms the 3D surfaces into 2D lines.

.. code-block:: python

    import pyvista as pv
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    # Open geometry file
    dataset = pv.read('geometry_painter.vtp')

    # Produce a slice along the x-z axis
    mesh = dataset.slice(normal=[0,1,0])

    # Draw lines in each slice cell
    plt.figure()
    xlim, ylim = (0,0), (0,0)
    for ind in range(mesh.n_cells):
        x, y, z = mesh.cell_points(ind).T

        if mesh.cell_type(ind) == 3:  # VTK_LINE
            line = mpl.lines.Line2D(z, x, lw=2, c='k')
            plt.gca().add_artist(line)
            xlim = (min(xlim[0],z.min()), max(xlim[1],z.max()))
            ylim = (min(ylim[0],x.min()), max(ylim[1],x.max()))
    plt.xlim(xlim)
    plt.ylim(ylim)


Kassiopieia visualization
-------------------------

The *Kassiopieia* module provides a set of stand-alone visualization tools that are described under :ref:`tools`.
The user may also specify visualization elements in the configuration file, which may be combined with the viewers
provided by *KGeoBag*. In fact this is often needed, if one wants to see e.g. the simulated trajectories within
the simulation geometry.

Below is an example that combines the VTK_ geometry painter of *KGeoBag* with a visualization of the simulated tracks
(``vtk_track_painter``) and the track terminator positions (``vtk_track_terminator_painter``). Note that in order
to use visualizations of simulation data, a ROOT_ output file has to exist.

.. code-block:: xml

    <vtk_window
            name="vtk_window"
            enable_display="true"
            enable_write="true"
            frame_title="Kassiopeia Visualization"
        >
        <vtk_geometry_painter
            name="geometry_painter"
            path="[output_path]"
            file="DipoleTrapGeometry.vtp"
            surfaces="world/dipole_trap/#"
        />
        <vtk_track_painter
                name="track_painter"
                path="[output_path]"
                file="DipoleTrapSimulation.root"
                point_object="component_step_world"
                point_variable="position"
                color_object="component_step_world"
                color_variable="polar_angle_to_b"
        />
        <vtk_track_terminator_painter
                name="terminator_painter"
                path="[output_path]"
                file="DipoleTrapSimulation.root"
                point_object="component_track_world"
                point_variable="final_position"
                terminator_object="component_track_world"
                terminator_variable="terminator_name"
                add_terminator="term_upstream_target"
                add_color="0 255 0"
                add_terminator="term_downstream_target"
                add_color="0 255 0"
                add_terminator="term_max_steps"
                add_color="255 0 0"
        />
    </vtk_window>

The options ``enable_display`` and ``enable_write`` of the ``<vtk_window>`` element specify if a viewer window should be
shown, and if an output file should be written. The output files can be viewed e.g. in the ParaView_ software. There
also exists a ``<vtk_generator_painter>`` element that is intended to visualize configured generators in the simulation.

A similar 2D visualization can be achieved using the ROOT_ visualization elements. The example below will present a view
of the 3D geometry projected onto the Z-X plane.

.. code-block:: xml

    <root_window
        name="Kassiopeia Visualization"
    >
        <root_pad name="toppad" xlow="0.02" ylow="0.98" xup="0.98" yup="0.98">
            <root_geometry_painter
                name="root_geometry_painter"
                surfaces="world/dipole_trap/#"
                plane_normal="0 1 0"
                plane_point="0 0 0"
                swap_axis="false"
            />
            <root_track_painter
                name="root_track_painter"
                path="[output_path]"
                base="DipoleTrapSimulation.root"
                plane_normal="0 1 0"
                plane_point="0 0 0"
                swap_axis="false"
                x_axis="z"
                y_axis="x"
                step_output_group_name="component_step_world"
                position_name="position"
                color_mode="track"
                color_variable="track_id"
            />
        </root_pad>
    </root_window>

It is possible to combine multiple such views into a single window by using the ``<root_pad>`` elements with
corresponding parameters. The projection mode has to be adjusted for the individual geometry painters. Another element,
``<root_zonal_harmonic_painter>``, can visualize the convergence radius and source points of the zonal harmonic
approximation that can be used for electric and magnetic field solving.


KEMField visualization
----------------------

The *KEMField* modules provides a special visualization that is only available for electrostatic geometries. In contrast
to the geometry viewers from *KGeoBag*, the *KEMField* viewer also includes extra information about the mesh elements,
the applied electric potentials, and the calculated charge densities. It is therefore extremely valuable for the design
of such geometries.

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

.. _ROOT: https://root.cern.ch/
.. _VTK: http://www.vtk.org/
.. _ParaView: https://www.paraview.org/
.. _PyVista: https://www.pyvista.org/
