Simulation
===========

The final object to be declared within ``<kassiopeia>`` is the simulation element. This describes the simulation object
**KSSimulation**, which is then executed by **KSRoot**. The simulation element specifies the global
and initial properties of the simulation as a whole. For example in the ``QuadrupoleTrapSimulation.xml`` example, the
simulation element is declared as follows:

.. code-block:: xml

    <ks_simulation
        name="quadrupole_trap_simulation"
        run="1"
        seed="51385"
        events="10"
        magnetic_field="field_electromagnet"
        electric_field="field_electrostatic"
        space="space_world"
        generator="generator_uniform"
        trajectory="trajectory_exact"
        space_navigator="nav_space"
        surface_navigator="nav_surface"
        writer="write_root"
    />

The ``run`` is simply a user provided identifier. The ``seed`` is the value provided to the global (singleton) random
number generator. Simulations with the same configuration and same seed should provide identical results. If the user is
interested in running *Kassiopeia* on many machines in order to achieve high throughput particle tracking (Monte Carlo),
care must be taken to ensure that the ``seed`` value is different for each run of the simulation.

The parameter ``events`` determines the total number of times that the generator is run (however this is not necessarily
the number of particles that will be tracked, e.g. if lists or sets are used in the generator of if secondary particles
are created). The remaining parameters ``magnetic_field``, ``space``, ``generator``, etc. all specify the default
objects to be used for the initial state of the simulation; where commands specified within ``ksgeo_space`` may
modify the actual objects used during the course of a simulation.

Following the declaration of ``ks_simulation``, the closing tag ``</kassiopeia>`` is placed to complete the simulation
configuration. When this tag is encountered by the XML parser, it triggers the simulation to run.

