Configuring Your Own Simulation
================================

.. _configuration-label:

The configuration of *Kassiopeia* is done in three main sections, plus an optional fourth. These are the ``<messages>``
section which describes global message verbosity, the ``<geometry>`` section (see :ref:`KGeoBag <KGeoBag>`) which describes the system geometry and its
extensions, the ``<kemfield>`` section (see :ref:`KEMField <KEMField>`) which defines the electromagnetic field elements, and the ``<kassiopeia>``
section (see :ref:`Kassiopeia element <Kassiopeia-element>`) which contains the elements needed for the particle tracking simulation. The optional fourth section relates to
the specification of a VTK_ or ROOT_ window for the visualization of aspects of a completed simulation. A complete
simulation file can thus be broken down into something which looks like the following:

.. code-block:: xml

    <messages>
        <!-- specification of the messaging configuration here -->
    </messages>

    <geometry>
        <!-- specification of the geometry and geometry extensions here -->
    </geometry>

    <kemfield>
        <!-- specification of the field elements and parameters here -->
    </kemfield>

    <kassiopeia>
        <!-- specification of the simulation elements and parameters here -->
    </kassiopeia>

    <!-- optional VTK window (requires VTK) -->
    <vtk_window>
        <!-- specification of the VTK window display here -->
    </vtk_window>

    <!-- optional ROOT window (requires ROOT) -->
    <root_window>
        <!-- specification of a ROOT display window here -->
    </root_window>

The XML based interface of *Kassiopeia* allows for a very flexible way in which to modify and configure particle
tracking simulations. Throughout the geometry, field, and simulation configuration all physical quantities are specified
using MKS_ and their derived units. The exception to this rule is energy, which is specified using electron volts (eV).
It should be noted that XML elements are parsed in order, and elements which are referenced by other elements must be
declared before their first use.

.. note::

    To test your configuration, you can use the given visualization techniques for each *Kassiopeia* section, which are described in detail in the following chapters.



.. toctree::
     :maxdepth: 1
     :hidden:

     Chapter overview <self>

.. toctree::
     :maxdepth: 1
     :caption: Chapter contents

     configuring_simulation_statements
     configuring_simulation_expressions
     configuring_simulation_messagingsystem
     configuring_simulation_pitfalls

