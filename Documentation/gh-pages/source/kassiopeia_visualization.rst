
.. _kassiopeia-visualization:

Visualization
==============

The *Kassiopieia* module provides a set of stand-alone visualization tools that are described under :ref:`tools-label`.
The user may also specify visualization elements in the configuration file, which may be combined with the viewers
provided by *KGeoBag*. In fact this is often needed, if one wants to see e.g. the simulated trajectories within
the simulation geometry.


Using VTK
~~~~~~~~~

Below is an example that combines the VTK_ geometry painter of *KGeoBag* with a visualization of the simulated tracks
(``vtk_track_painter``) and the track terminator positions (``vtk_track_terminator_painter``). 

.. note::

    In order to use visualizations of simulation data, a ROOT_ output file has to exist.


.. code-block:: xml

    <vtk_window
            name="vtk_window"
            enable_display="true"
            enable_write="true"
            frame_title="KGeoBag Visualization"
            frame_size_x="1024"
            frame_size_y="768"
            frame_color_red=".2"
            frame_color_green=".2"
            frame_color_blue=".2"
            view_angle="45"
            eye_angle="0.5"
            multi_samples="4"
            depth_peeling="10"
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
By adding the ``vtk_window`` element at the end of the configuration file, we activate a VTK window that will open when
the simulation is complete.

.. note::

    The visualization window must be placed outside of the ``<kassiopeia>``...``</kassiopiea>`` environment tags.

Using ROOT
~~~~~~~~~~~~

A similar 2D visualization can be achieved using the ROOT_ visualization elements. 
In constrast to VTK_, which displays three-dimensional geometry, the ROOT_ visualization is limited to two dimensions. 
The example below will present a view of the 3D geometry projected onto the Z-X plane.

.. code-block:: xml

    <root_window
        name="Kassiopeia Visualization"
        canvas_width="1000"
        canvas_height="600"
        active="active"
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

Using Python
~~~~~~~~~~~~

The track painters export VTK_ output files that can be visualized in Python with the PyVista_ module, as shown in :ref:`kgeobag-visualization`.



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
.. _Gnuplot: http://www.gnuplot.info/
.. _PyVista: https://www.pyvista.org/

