.. _kgeobag-visualization:

KGeoBag Visualization
=====================

The *KGeoBag* module provides a set of stand-alone visualization tools that are described under :ref:`tools-label`. These
are suited to display the simulation geometry and other geometric elements, such as the mesh used for field calculation
and/or navigation.

In addition, the geometry visualization can also be defined in the configuration file. In this case, output files
may be produced before or after performing the simulation, and a visualization window can be shown as well. Note that
the visualization window blocks the application until it is closed, so it is not advised to use this feature in a
scripted environment. The examples in this section are based on the ``DipoleTrapSimulation.xml`` file, which may be extended accordingly to test the features explained here and in the following chapters. 
The commandline option ``-b`` (or ``-batch``) will prevent any visualization windows to appear regardless of the setting in the configuration file, e.g.:

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

The axial mesh painter needs a defined mesh (``<axial_mesh>`` XML element, see :ref:`Configuring Your Own Simulation <configuration-label>`). An
``<vtk_mesh_painter>`` exists as well, to be used with an asymmetric mesh (defined via ``<mesh>``).

**Using Python**


It is possible to draw a geometry visualization in Python. This is especially useful if you run your analysis in Python as well (see :ref:`output-label` for examples.)

The PyVista_ Python package makes it easy to operate on the VTK_ output files that are produced by *KGeoBag* and the 
other *Kasper* modules. (Note that this method only works if VTK_ was enabled at build time.) In the XML snippets above, 
the different VTK painters will produce output files if the ``enable_write`` attribute is set. These files contain the 
3D geometry in the `vtp` format and can be read in Python. In order to get a 2D view of the geometry, one also needs to 
create a slice that transforms the 3D surfaces into 2D lines.

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
