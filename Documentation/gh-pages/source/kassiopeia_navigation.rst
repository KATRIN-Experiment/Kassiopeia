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

The space navigator is more complex since it is responsible for determining the location of the particle and any possible
intersections it may have with real or virtual boundaries. It is also responsible for ensuring that the current
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

.. xml-element:: ksnav_meshed_space

Commands
----------

For dynamic configuration, *Kassiopeia* allows certain commands to be used during the calculation of a particle
trajectory. The commands are associated with particular surfaces and spaces and are what effectively governs the state
of the simulation as a particle is tracked. They are declared through the specification of a ``ksgeo_space``. A very
simple example of the declaration of the command structure can be seen in the DipoleTrapSimulation.xml as shown below:

.. code-block:: xml

    <ksgeo_space name="space_world" spaces="world">
        <add_terminator parent="root_terminator" child="term_max_steps"/>
        <remove_terminator parent="root_terminator" child="term_world"/>
        <add_track_output parent="write_root" child="component_track_world"/>
        <add_step_output parent="write_root" child="component_step_world"/>

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

which are equivalent to the commands shown above.

For commands that act on one of the root objects (terminators, electric and magnetic fields, space and surface
interactions, step modifiers and the trajectory), the ``parent`` attribute may be omitted; the corresponding root object
is then used as parent, e.g. ``root_terminator`` for ``add_terminator``. All other commands require an explicit
``parent``.

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

Commands can be used to change the active field calculation method, swap trajectory types, or add/remove terms and
interactions, define terminators, etc. Various spaces and their associated commands can be nested within each other
allowing for a very flexible and dynamic simulation configuration. For best results, it is important that the
structure of the ``geo_space`` and ``geo_surface`` elements follows the structure of the *KGeoBag* geometry tree, i.e.
nesting of the navigation elements should follow the same order as the underlying geometry.



