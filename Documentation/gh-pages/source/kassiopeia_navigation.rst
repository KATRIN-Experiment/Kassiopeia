Navigation
===========

Surfaces navigation
~~~~~~~~~~~~~~~~~~~

The navigation of a particle is split into two components, surface navigation and space navigation. Surface navigation
is very simple and only takes place when a particle has intersected an active surface. The surface navigator determines
whether the state of the particle is modified on the surface and whether it is reflected or transmitted. It can be made
available for use with the declaration:

.. code-block:: xml

    <ksnav_surface name="nav_surface" transmission_split="false" reflection_split="false"/>

As this navigator is very simple, it does not take many additional parameters. The parameters ``transmission_split`` and
``reflection_split`` determine whether or not a track is split in two (for the purposes of output/saving data) upon
transmission through or reflection off a geometric surface.

Space navigation
~~~~~~~~~~~~~~~~

The space navigator is more complex since it is responsible for determine the location of the particle and any possible
intersections it may have with real of virtual boundaries. It is also responsible for ensuring that the current
simulation state matches the configuration specified by the user. The spaces that the navigator considers may be real
objects (such as a vacuum chamber wall) or virtual (which only modify the state of the simulation, e.g. exchanging a
trajectory method). The latter case allows to dynamically reconfigure the simulation over a track.

For space navigation there are two options. The first is the default ``ksnav_space`` navigator which can be specified as
below:

.. code-block:: xml

    <ksnav_space name="nav_space" enter_split="false" exit_split="false"/>

As this navigator is also very simple, it does not take many additional parameters. The parameters ``enter_split`` and
``exit_split`` determine whether or not a track is split in two  upon entrance or exit of a geometric space.

Meshed space navigation
~~~~~~~~~~~~~~~~~~~~~~~

A more complex behavior is achieved by the ``ksnav_meshed_space`` navigator, which is intended to be used in highly
detailed three-dimensional geometries where it has better performance over the default navigator. An example of this is
shown in the ``PhotoMultplierTube.xml`` file. Its configuration is more complicated as it performs the navigations on
the meshed boundaries of spaces and surfaces. It requires the construction of an octree spatial partition (which may be
cached for later re-use). The user must specify the depth of the octree (``max_octree_depth``) and the number of
elements allowed in a octree node before a subdivision is required (``n_allowed_elements``). In addition, the root of
the geometry tree must also be specified with the parameter ``root_space``, which is typically the world volume:

.. code-block:: xml

    <ksnav_meshed_space name="nav_meshed_space" root_space="space_world" max_octree_depth="9" n_allowed_elements="1"/>

Though they are not shown here (they default to false), the exit and entrance split parameters may also be set for the
``ksnav_meshed_space`` navigator. Because the ``ksnav_meshed_space`` navigator requires a boundary mesh in order to
operate, all geometric objects (spaces, sufaces) which have navigation commands attached to them must also have a mesh
extension in the geometry specification. Furthermore, since ``ksnav_meshed_space`` requires access to the root space
``space_world`` and all of the navigation commands associated with the shapes it contains, it must be declared after the
definition of the simulation command structure element ``ksgeo_space`` (see below).

The mesh navigator can also be used together with geometry from exernal files, as shown in the ``MeshSimulation.xml``
example.

Commands
----------

For dyanmic configuration, *Kassiopeia* allows certain commands to be used during the calculation of a particle
trajectory. The commands are associated with particular surfaces and spaces and are what effectively governs the state
of the simulation as a particle is tracked. They are declared through the specification of a ``ksgeo_space``. A very
simple example of the declaration of the command structure can be seen in the DipoleTrapSimulation.xml as shown below:

.. code-block:: xml

    <ksgeo_space name="space_world" spaces="world">
        <add_terminator parent="root_terminator" child="term_max_steps"/>
        <remove_terminator parent="root_terminator" child="term_world"/>
        <add_track_output parent="write_root" child="component_track_world"/>
        <comadd_step_outputmand parent="write_root" child="component_step_world"/>

        <geo_surface name="surface_upstream_target" surfaces="world/dipole_trap/upstream_target">
            <add_terminator parent="root_terminator" child="term_upstream_target"/>
        </geo_surface>

        <geo_surface name="surface_downstream_target" surfaces="world/dipole_trap/downstream_target">
            <add_terminator parent="root_terminator" child="term_downstream_target"/>
        </geo_surface>

        <geo_surface name="surface_center" surfaces="world/dipole_trap/center"/>
    </ksgeo_space>

Note that in some configuration files, you may find alternative declarations such as:

.. code-block:: xml

    <command parent="root_terminator" field="add_terminator" child="term_max_steps"/>
    <command parent="root_terminator" field="add_terminator" child="term_upstream_target"/>

which are eequivalent to the commands shown above.

Again, let us break down this example:

- First we create a ``ksgeo_space`` navigation space using the ``world`` volume (a geometric object holding all other
  elements.) Inside of this world volume we declare a series of command which will be executed any time a particle
  enters or is initialized within the world volume.
- The first two commands add and remove specific terminators, while the next two declare what sort of output should be
  written to disk while the particle is inside the world volume.
- Following that, there are commands which are attached to specific surfaces which are present in the geometry, and
  handled by the navigator. For example in the first block, attaching the terminator ``term_upstream_target`` ensures
  that a particle impinging on the surface ``surface_upstream_target`` will be terminated immediately.
- The last surface does not have any associated commands, but will still be considered for navigation. For example,
  if ``transmission_split`` was set in the navigator, the track will be split if the particle crosses the surface.

Commands can used to change the active field calculation method, swap trajectory types, or add/remove termsna and
interactions, define terminators, etc. Various spaces and their associated commands can be nested within each other
allowing for a very flexible and dynamic simulation configuration. For best results, it is important that the
structure of the ``geo_space`` and ``geo_surface`` elements follows the structure of the *KGeoBag* geometry tree, i.e.
nesting of the navigation elements should follow the same order as the underlying geometry.



