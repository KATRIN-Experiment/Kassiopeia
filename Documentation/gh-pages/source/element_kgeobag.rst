.. _KGeoBag:

KGeoBag (geometry) - <geometry>
===============================

The geometry section of the configuration file is the first piece needed in order to assemble a simulation. At its first
and most basic level it is responsible for defining all the different shapes that will be used, and placing them with
respect to one another in order to construct the arrangement (often called `assembly`) that is needed.

A full a description of all of the shape objects (surfaces and spaces) which maybe constructed in *KGeoBag* can be found in this chapter. 
The abstract base classes which serve as the interface between *KGeoBag* and *Kassiopeia* are ``KSSpace``, 
``KSSurface``, and ``KSSide`` (see :gh-code:`Kassiopeia/Operators`).

The geometry section is also responsible for adding "extended" information to the defined geometry elements. These
extensions can be properties such as colors for visualization, or boundary conditions and meshing details for the
electromagnetic simulations.

Every relevant to the geometry description is processed by *KGeoBag* and must appear between the start and end brackets:

.. code-block:: xml

    <geometry>
        <!-- fill in geometry description here -->
    </geometry>

.. note::

    The full description of the geometry need not lie within the same pair of ``<geometry>`` and ``</geometry>`` brackets. 
    This facilitates the description of separate geometry pieces in different files, which may
    then be included and used in the final assembly.

.. toctree::
     :maxdepth: 1
     :hidden:

     Chapter overview <self>

.. toctree::
     :maxdepth: 1
     :caption: Chapter contents
     
     kgeobag_geometry
     kgeobag_simple_shapes
     kgeobag_complex_shapes
     kgeobag_visualization
     kgeobag_tools
