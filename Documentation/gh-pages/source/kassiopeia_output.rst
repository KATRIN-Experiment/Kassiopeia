Output
========


The data which is saved as output from the simulation requires two pieces: a file writer and a description of the data
to be saved. The abstract base class of all file writers is **KSWriter**.

Writers
~~~~~~~

The file writer is responsible for buffering and writing the desired information to disk. The default writer is based on
ROOT_, and stores the output in a ``TTree`` structure:

.. code-block:: xml

    <kswrite_root
        name="write_root"
        path="/path/to/desired/output/directory"
        base="my_filename.root"
    />

If *Kassiopeia* is linked against VTK_, an additional writer will be made available which can save track and step
information to a ``.vtp`` (VTK polydata) file. This data is useful for visualalization in external tools such as
Paraview_. This write may be created using the following statement:

.. code-block:: xml

    <kswrite_vtk
        name="write_vtk"
        path="/path/to/desired/output/directory"
        base="my_filename_base.vtp"
    />

Note that in principle both data formats are equivalent, but their underlying structure differs. In most cases it is
best to write output file in both formats, and delete any files that are no longer needed.

To write output in plaintext ASCII format that can be easily viewed and read into other software such as Gnuplot_,
one may use the following statement:

.. code-block:: xml

    <kswrite_ascii
        name="write_ascii"
        path="/path/to/desired/output/directory"
        base="my_filename_base.vtp"
    />

This is not recommended for large-scale simulations because the output file will quickly approach a size that will be
extremely difficult to handle.

Output description
~~~~~~~~~~~~~~~~~~

The user may tailor the data written to disk to keep precisely the quantities of interest and no more. To do this a
description of the data components to be kept at the track and step level must be given. An example of this (taken from
the ``QuadrupoleTrapSimulation.xml`` example) is shown below:

.. code-block:: xml

    <ks_component_member name="component_step_final_particle" field="final_particle" parent="step"/>
    <ks_component_member name="component_step_position" field="position" parent="component_step_final_particle"/>
    <ks_component_member name="component_step_length" field="length" parent="component_step_final_particle"/>

    <ks_component_group name="component_step_world">
        <component_member name="step_id" field="step_id" parent="step"/>
        <component_member name="continuous_time" field="continuous_time" parent="step"/>
        <component_member name="continuous_length" field="continuous_length" parent="step"/>
        <component_member name="time" field="time" parent="component_step_final_particle"/>
        <component_member name="position" field="position" parent="component_step_final_particle"/>
        <component_member name="momentum" field="momentum" parent="component_step_final_particle"/>
        <component_member name="magnetic_field" field="magnetic_field" parent="component_step_final_particle"/>
        <component_member name="electric_field" field="electric_field" parent="component_step_final_particle"/>
        <component_member name="electric_potential" field="electric_potential" parent="component_step_final_particle"/>
        <component_member name="kinetic_energy" field="kinetic_energy_ev" parent="component_step_final_particle"/>
    </ks_component_group>

    <ks_component_group name="component_step_cell">
        <component_member name="polar_angle_to_z" field="polar_angle_to_z" parent="component_step_final_particle"/>
        <component_member name="polar_angle_to_b" field="polar_angle_to_b" parent="component_step_final_particle"/>
        <component_member name="guiding_center_position" field="guiding_center_position" parent="component_step_final_particle"/>
        <component_member name="orbital_magnetic_moment" field="orbital_magnetic_moment" parent="component_step_final_particle"/>
    </ks_component_group>

    <ks_component_member name="component_track_initial_particle" field="initial_particle" parent="track"/>
    <ks_component_member name="component_track_final_particle" field="final_particle" parent="track"/>
    <ks_component_member name="component_track_position" field="position" parent="component_track_final_particle"/>
    <ks_component_member name="component_track_length" field="length" parent="component_track_final_particle"/>

    <ks_component_member name="z_length" field="continuous_length" parent="step"/>
    <ks_component_group name="component_track_world">
        <component_member name="creator_name" field="creator_name" parent="track"/>
        <component_member name="terminator_name" field="terminator_name" parent="track"/>
        <component_member name="total_steps" field="total_steps" parent="track"/>
        <component_member name="initial_time" field="time" parent="component_track_initial_particle"/>
        <component_member name="initial_position" field="position" parent="component_track_initial_particle"/>
        <component_member name="initial_momentum" field="momentum" parent="component_track_initial_particle"/>
        <component_member name="initial_magnetic_field" field="magnetic_field" parent="component_track_initial_particle"/>
        <component_member name="initial_electric_field" field="electric_field" parent="component_track_initial_particle"/>
        <component_member name="initial_electric_potential" field="electric_potential" parent="component_track_initial_particle"/>
        <component_member name="initial_kinetic_energy" field="kinetic_energy_ev" parent="component_track_initial_particle"/>
        <component_member name="initial_polar_angle_to_z" field="polar_angle_to_z" parent="component_track_initial_particle"/>
        <component_member name="initial_azimuthal_angle_to_x" field="azimuthal_angle_to_x" parent="component_track_initial_particle"/>
        <component_member name="initial_polar_angle_to_b" field="polar_angle_to_b" parent="component_track_initial_particle"/>
        <component_member name="initial_orbital_magnetic_moment" field="orbital_magnetic_moment" parent="component_track_initial_particle"/>
        <component_member name="final_time" field="time" parent="component_track_final_particle"/>
        <component_member name="final_position" field="position" parent="component_track_final_particle"/>
        <component_member name="final_momentum" field="momentum" parent="component_track_final_particle"/>
        <component_member name="final_magnetic_field" field="magnetic_field" parent="component_track_final_particle"/>
        <component_member name="final_electric_field" field="electric_field" parent="component_track_final_particle"/>
        <component_member name="final_electric_potential" field="electric_potential" parent="component_track_final_particle"/>
        <component_member name="final_kinetic_energy" field="kinetic_energy_ev" parent="component_track_final_particle"/>
        <component_member name="final_polar_angle_to_z" field="polar_angle_to_z" parent="component_track_final_particle"/>
        <component_member name="final_azimuthal_angle_to_x" field="azimuthal_angle_to_x" parent="component_track_final_particle"/>
        <component_member name="final_polar_angle_to_b" field="polar_angle_to_b" parent="component_track_final_particle"/>
        <component_member name="final_orbital_magnetic_moment" field="orbital_magnetic_moment" parent="component_track_final_particle"/>
        <component_member name="z_length_internal" field="continuous_length" parent="track"/>
        <component_integral name="z_length_integral" parent="z_length"/>
    </ks_component_group>

Let us break this down a bit. First of all, the output can be separated into three groups that each define an output
segment that will be written to the file:

- `component_step_world` is the base definition for output at the step level. It contains standard parameters of the
  particle such as its energy, position, or step index.
- `component_step_cell` defines additional output fields that are of interest in a specific region of the simulation.
  How this feature can be used will be explained below. Generally, one can define as many output groups as necessary
  to write output only where it is relevant to the simulation.
- `component_track_world` is the base definition for output at the track level. While the step output is written
  continuously while the particle trajectory is being computed, the track output is only written once after a track
  has been terminated. As such, the track output contains initial and final parameters of the particle (again, for
  example, its energy or position) and are derived from the first and last step of the track. There is also an output
  field ``z_length_integral`` that stores the integrated length of all tracks performed in the simulation.

For output fields that are not directly available at the step (``parent="step"``) or track level, a mapping has to be
defined first. This is done by the lines:

.. code-block:: xml

    <ks_component_member name="component_step_final_particle" field="final_particle" parent="step"/>

and so on. The ``field="final_particle"`` points to the final particle state after a step has been performed, i.e. this
output is written after the completion of each step. Similary, at the track level there are output fields that point
to the initial and final parameters of a track, i.e. the state at particle generation and termination.

The standard output fields for the particle are defined at the end of the file
:gh-code:`Kassiopeia/Operators/Source/KSParticle.cxx` while the step and track output fields can be found in
:gh-code:`Kassiopeia/Operators/Source/KSStep.cxx` and :gh-code:`Kassiopeia/Operators/Source/KSTrack.cxx`, respectively.
Other specialized output fields are also available for some propagation or interaction terms.

Output fields
~~~~~~~~~~~~~

Many different output fields can be used and combined in the output configuration. The table below gives an
overview of the different fields and their types.

+---------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Output fields                                                                                                                                                       |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Name               | XML Element                         | Value Type       | Base class                 |  Description (main parameters)                           |
+====================+=====================================+==================+============================+==========================================================+
| Index Number       | ``index_number``                    | ``long``         | ``KSParticle``             | Unique index number of the current step                  |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Parent Run ID      | ``parent_run_id``                   | ``int``          | ``KSParticle``             | Run ID of the parent step/track/event                    |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Parent Event ID    | ``parent_event_id``                 | ``int``          | ``KSParticle``             | Event ID of the parent step/track/event                  |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Parent Track ID    | ``parent_track_id``                 | ``int``          | ``KSParticle``             | Track ID of the parent step/track                        |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Parent Step ID     | ``parent_step_id``                  | ``int``          | ``KSParticle``             | Step ID of the parent step                               |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Particle ID        | ``pid``                             | ``long long``    | ``KSParticle``             | Assigned particle ID (PDG code)                          |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Particle String ID | ``string_id``                       | ``string``       | ``KSParticle``             | Assigned particle ID (human-readable)                    |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Particle Mass      | ``mass``                            | ``double``       | ``KSParticle``             | Mass of the particle (in kg)                             |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Particle Charge    | ``charge``                          | ``double``       | ``KSParticle``             | Charge of the particle (in C)                            |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Particle Spin      | ``total_spin``                      | ``double``       | ``KSParticle``             | Spin magnitude of the particle (in hbar)                 |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Gyromagnetic Ratio | ``gyromagnetic_ratio``              | ``double``       | ``KSParticle``             | Gyromagnetic ratio of the particle (in rad/sT)           |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Main Quantum No.   | ``n``                               | ``int``          | ``KSParticle``             | Main quantum number                                      |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Second Quatum No.  | ``l``                               | ``int``          | ``KSParticle``             | Secondary quantum number                                 |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Time               | ``time``                            | ``double``       | ``KSParticle``             | Time in the simulation (in s)                            |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Wallclock Time     | ``clock_time``                      | ``double``       | ``KSParticle``             | Wallclock time (system time) at the current step         |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Step Length        | ``length``                          | ``double``       | ``KSParticle``             | Length of the current step (in m)                        |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Position Vector    | ``position``                        | ``KThreeVector`` | ``KSParticle``             | Position at the current step (in m)                      |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Momentum Vector    | ``momentum``                        | ``KThreeVector`` | ``KSParticle``             | Momentum at the current step (in kg*m/s)                 |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Velocity Vector    | ``velocity``                        | ``double``       | ``KSParticle``             | Velocity at the current step (in m/s)                    |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Spin Vector        | ``spin``                            | ``KThreeVector`` | ``KSParticle``             | Spin at the current step (in hbar)                       |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Index Number       | ``spin0``                           | ``double``       | ``KSParticle``             |                                                          |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Aligned Spin       | ``aligned_spin``                    | ``double``       | ``KSParticle``             |                                                          |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Spin Angle         | ``spin_angle``                      | ``double``       | ``KSParticle``             |                                                          |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Speed              | ``speed``                           | ``double``       | ``KSParticle``             | Total speed at the current step (in m/s)                 |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Lorentz Factor     | ``lorentz_factor``                  | ``double``       | ``KSParticle``             | Lorentz factor at the current step                       |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Kinetic Energy     | ``kinetic_energy``                  | ``double``       | ``KSParticle``             | Kinetic energy at the current step (in J)                |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Kinetic Energy     | ``kinetic_energy_ev``               | ``double``       | ``KSParticle``             | Kinetic energy at the current step (in eV)               |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Polar Angle        | ``polar_angle_to_z``                | ``double``       | ``KSParticle``             | Polar angle relative to z-axis (in deg)                  |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Azimuthal Angle    | ``azimuthal_angle_to_x``            | ``double``       | ``KSParticle``             | Azimuthal angle relative to x-axis (in deg)              |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Magnetic Field     | ``magnetic_field``                  | ``KThreeVector`` | ``KSParticle``             | Magnetic field at the current step (in T)                |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Electric Field     | ``electric_field``                  | ``KThreeVector`` | ``KSParticle``             | Electric field at the current step (in V/m)              |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Magnetic Gradient  | ``magnetic_gradient``               | ``KThreeMatrix`` | ``KSParticle``             | Magnetic gradient at the current step (in T/m)           |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Electric Potential | ``electric_potential``              | ``double``       | ``KSParticle``             | Electric potential at the current step (in V)            |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Long. Momentum     | ``long_momentum``                   | ``double``       | ``KSParticle``             | Longitudinal momentum at the current step (in kg*m/s)    |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Trans. Momentum    | ``trans_momentum``                  | ``double``       | ``KSParticle``             | Transversal momentum at the current step (in kg*m/s)     |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Long. Velocity     | ``long_velocity``                   | ``double``       | ``KSParticle``             | Longitudinal velocity at the current step (in m/s)       |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Trans. Velocity    | ``trans_velocity``                  | ``double``       | ``KSParticle``             | Transversal velocity at the current step (in m/s)        |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Polar Angle to B   | ``polar_angle_to_b``                | ``double``       | ``KSParticle``             | Polar (pitch) angle relative to magnetic field (in deg)  |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Cyclotron Freq.    | ``cyclotron_frequency``             | ``double``       | ``KSParticle``             | Cyclotron frequency at the current step (in Hz)          |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Magnetic Moment    | ``orbital_magnetic_moment``         | ``double``       | ``KSParticle``             | Orbital magnetic moment at the current step (in A*m^2)   |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| GC Position Vector | ``guiding_center_position``         | ``KThreeVector`` | ``KSParticle``             | Guiding center position at the current step (in m)       |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Current Space      | ``current_space_name``              | ``string``       | ``KSParticle``             | Name of the nearest space (see ``geo_space``)            |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Current Surface    | ``current_surface_name``            | ``string``       | ``KSParticle``             | Name of the nearest surface (see ``geo_surface``)        |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Current Side       | ``current_side_name``               | ``string``       | ``KSParticle``             | Name of the nearest side (see ``geo_side``)              |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| GC Velocity        | ``gc_velocity``                     | ``double``       | ``KSTrajTermDrift``        | Guiding center velocity (in m/s)                         |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| GC Long. Force     | ``longitudinal_force``              | ``double``       | ``KSTrajTermDrift``        | Longitudinal force added by drift terms (in N)           |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| GC Trans. Force    | ``transverse_force``                | ``double``       | ``KSTrajTermDrift``        | Transversal force added by drift terms (in N)            |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Gy. Phase Velocity | ``phase_velocity``                  | ``double``       | ``KSTrajTermGyration``     | Phase velocity of gyration around g.c. (in rad/s)        |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Synchrotron Force  | ``total_force``                     | ``double``       | ``KSTrajTermSynchrotron``  | Total force added by synchrotron radiation (in N)        |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Min. Distance      | ``min_distance``                    | ``double``       | ``KSTermMinDistance``      | Distance to the nearest surface (in m)                   |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Interaction Count  | ``step_number_of_interactions``     | ``int``          | ``KSIntCalculator``        | Number of interactions  at current step                  |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Energy loss        | ``step_energy_loss``                | ``double``       | ``KSIntCalculator``        | Energy loss at current step (in eV)                      |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Angular Change     | ``step_angular_change``             | ``double``       | ``KSIntCalculator``        | Angular change at current step (in deg)                  |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Interaction Count  | ``step_number_of_decays``           | ``int``          | ``KSIntDecayCalculator``   | Number of interactions  at current step                  |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Energy loss        | ``step_energy_loss``                | ``double``       | ``KSIntDecayCalculator``   | Energy loss at current step (in eV)                      |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Enhancement Factor | ``enhancement_factor``              | ``double``       | ``KSModDynamicEnhancement``| Step modifier enhancement factor                         |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Run ID             | ``run_id``                          | ``int``          | ``KSRun``                  | Run ID of current run                                    |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Run Count          | ``run_count``                       | ``int``          | ``KSRun``                  | Total number of runs                                     |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Total Events       | ``total_events``                    | ``unsigned int`` | ``KSRun``                  | Total number of events in run                            |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Total Tracks       | ``total_tracks``                    | ``unsigned int`` | ``KSRun``                  | Total number of tracks in run                            |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Total Steps        | ``total_steps``                     | ``unsigned int`` | ``KSRun``                  | Total number of steps in run                             |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Cont. Time         | ``continuous_time``                 | ``double``       | ``KSRun``                  | Total time of all events/tracks/steps in run             |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Cont. Length       | ``continuous_length``               | ``double``       | ``KSRun``                  | Total length of all events/tracks/steps in run           |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Energy Change      | ``continuous_energy_change``        | ``double``       | ``KSRun``                  | Total energy change during run                           |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Momentum Change    | ``continuous_momentum_change``      | ``double``       | ``KSRun``                  | Total momentum change during run                         |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Secondaries Count  | ``discrete_secondaries``            | ``unsigned int`` | ``KSRun``                  | Number of secondaries created during run                 |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Energy Change      | ``discrete_energy_change``          | ``double``       | ``KSRun``                  | Total energy change during run                           |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Momentum Change    | ``discrete_momentum_change``        | ``double``       | ``KSRun``                  | Total momentum change during run                         |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Number of Turns    | ``number_of_turns``                 | ``unsigned int`` | ``KSRun``                  | Number of particle turns/reflections during run          |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Event ID           | ``event_id``                        | ``int``          | ``KSEvent``                | Event ID of current event                                |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Event Count        | ``event_count``                     | ``int``          | ``KSEvent``                | Total number of events                                   |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Parent Run ID      | ``parent_run_id``                   | ``int``          | ``KSEvent``                | Run ID of parent run                                     |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Total Tracks       | ``total_tracks``                    | ``unsigned int`` | ``KSEvent``                | Total number of tracks in event                          |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Total Steps        | ``total_steps``                     | ``unsigned int`` | ``KSEvent``                | Total number of steps in event                           |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Cont. Time         | ``continuous_time``                 | ``double``       | ``KSEvent``                | Total time of all tracks/steps in event                  |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Cont. Length       | ``continuous_length``               | ``double``       | ``KSEvent``                | Total length of all tracks/steps in event                |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Energy Change      | ``continuous_energy_change``        | ``double``       | ``KSEvent``                | Total energy change during event                         |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Momentum Change    | ``continuous_momentum_change``      | ``double``       | ``KSEvent``                | Total momentum change during event                       |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Secondaries Count  | ``discrete_secondaries``            | ``unsigned int`` | ``KSEvent``                | Number of secondaries created during event               |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Energy Change      | ``discrete_energy_change``          | ``double``       | ``KSEvent``                | Total energy change during event                         |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Momentum Change    | ``discrete_momentum_change``        | ``double``       | ``KSEvent``                | Total momentum change during event                       |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Number of Turns    | ``number_of_turns``                 | ``unsigned int`` | ``KSEvent``                | Number of particle turns/reflections during event        |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Generator Name     | ``generator_name``                  | ``string``       | ``KSEvent``                | Name of the generator starting this event                |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Generator Flag     | ``generator_flag``                  | ``bool``         | ``KSEvent``                | Additional flag of the used generator                    |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Primary Count      | ``generator_primaries``             | ``unsigned int`` | ``KSEvent``                | Number of generated particles                            |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Generator Energy   | ``generator_energy``                | ``double``       | ``KSEvent``                | Total energy of the generated particles (in eV)          |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Generator Time     | ``generator_min_time``              | ``double``       | ``KSEvent``                | Minimum time of the generated particles (in s)           |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Generator Time     | ``generator_max_time``              | ``double``       | ``KSEvent``                | Maximum time of the generated particles (in s)           |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Generator Position | ``generator_location``              | ``KThreeVector`` | ``KSEvent``                | Center position of the generated particles (in m)        |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Generator Radius   | ``generator_radius``                | ``double``       | ``KSEvent``                | Maximum radius of the generated particles (in m)         |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Track ID           | ``event_id``                        | ``int``          | ``KSTrack``                | Track ID of current track                                |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Track Count        | ``event_count``                     | ``int``          | ``KSTrack``                | Total number of tracks                                   |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Parent Event ID    | ``parent_event_id``                 | ``int``          | ``KSTrack``                | Event ID of parent track                                 |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Total Steps        | ``total_steps``                     | ``unsigned int`` | ``KSTrack``                | Total number of steps in track                           |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Cont. Time         | ``continuous_time``                 | ``double``       | ``KSTrack``                | Total time of all steps in track                         |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Cont. Length       | ``continuous_length``               | ``double``       | ``KSTrack``                | Total length of all steps in track                       |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Energy Change      | ``continuous_energy_change``        | ``double``       | ``KSTrack``                | Total energy change during track                         |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Momentum Change    | ``continuous_momentum_change``      | ``double``       | ``KSTrack``                | Total momentum change during track                       |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Secondaries Count  | ``discrete_secondaries``            | ``unsigned int`` | ``KSTrack``                | Number of secondaries created during track               |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Energy Change      | ``discrete_energy_change``          | ``double``       | ``KSTrack``                | Total energy change during track                         |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Momentum Change    | ``discrete_momentum_change``        | ``double``       | ``KSTrack``                | Total momentum change during track                       |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Number of Turns    | ``number_of_turns``                 | ``unsigned int`` | ``KSTrack``                | Number of particle turns/reflections during track        |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Creator Name       | ``creator_name``                    | ``string``       | ``KSTrack``                | Name of the creator starting this track                  |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Terminator Name    | ``terminator_name``                 | ``string``       | ``KSTrack``                | Name of the terminator ending this track                 |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Initial Particle   | ``initial_particle``                | ``KSParticle``   | ``KSTrack``                | Pointer to initial particle at begin of the track        |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Final particle     | ``final_particle``                  | ``KSParticle``   | ``KSTrack``                | Pointer to final particle at end of the track            |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Step ID            | ``step_id``                         | ``int``          | ``KSStep``                 | Step ID of current step                                  |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Step Count         | ``step_count``                      | ``int``          | ``KSStep``                 | Total number of steps                                    |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Parent Track ID    | ``parent_track_id``                 | ``int``          | ``KSStep``                 | Track ID of parent track                                 |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Cont. Time         | ``continuous_time``                 | ``double``       | ``KSStep``                 | Total time of current step                               |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Cont. Length       | ``continuous_length``               | ``double``       | ``KSStep``                 | Total length of current step                             |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Energy Change      | ``continuous_energy_change``        | ``double``       | ``KSStep``                 | Total energy change during step                          |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Momentum Change    | ``continuous_momentum_change``      | ``double``       | ``KSStep``                 | Total momentum change during step                        |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Secondaries Count  | ``discrete_secondaries``            | ``unsigned int`` | ``KSStep``                 | Number of secondaries created during step                |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Energy Change      | ``discrete_energy_change``          | ``double``       | ``KSStep``                 | Total energy change during step                          |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Momentum Change    | ``discrete_momentum_change``        | ``double``       | ``KSStep``                 | Total momentum change during step                        |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Number of Turns    | ``number_of_turns``                 | ``unsigned int`` | ``KSStep``                 | Number of particle turns/reflections during step         |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Modifier Name      | ``modifier_name``                   | ``string``       | ``KSStep``                 | Name of the step modifier at this step                   |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Modifier Flag      | ``modifier_flag``                   | ``bool``         | ``KSStep``                 | Additional flag for the used terminator                  |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Terminator Name    | ``terminator_name``                 | ``string``       | ``KSStep``                 | Name of the terminator ending this step                  |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Terminator Flag    | ``terminator_flag``                 | ``bool``         | ``KSStep``                 | Additional flag for the used terminator                  |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Trajectory Name    | ``trajectory_name``                 | ``string``       | ``KSStep``                 | Name of the trajectory at this step                      |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Trajectory Center  | ``trajectory_center``               | ``KThreeVector`` | ``KSStep``                 | Position of the trajectory bounding sphere (in m)        |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Trajectory Radius  | ``trajectory_radius``               | ``double``       | ``KSStep``                 | Radius of the trajectory bounding sphere (in m)          |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Trajectory Step    | ``trajectory_step``                 | ``double``       | ``KSStep``                 | Time of the particle propagation                         |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Interaction Name   | ``space_interaction_name``          | ``string``       | ``KSStep``                 | Space name of the interaction at this step               |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Interaction Flag   | ``space_interaction_flag``          | ``bool``         | ``KSStep``                 | Additional flag for the space interaction                |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Interaction Step   | ``space_interaction_step``          | ``double``       | ``KSStep``                 | Time of the space interaction                            |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Navigation Name    | ``space_navigation_name``           | ``string``       | ``KSStep``                 | Space name of the navigation at this step                |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Navigation Flag    | ``space_navigation_flag``           | ``bool``         | ``KSStep``                 | Additional flag for the space navigation                 |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Navigation Step    | ``space_navigation_step``           | ``double``       | ``KSStep``                 | Time of the space navigation                             |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Interaction Name   | ``surface_interaction_name``        | ``string``       | ``KSStep``                 | Surface name of the interaction at this step             |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Interaction Flag   | ``surface_interaction_flag``        | ``bool``         | ``KSStep``                 | Additional flag for the surface interaction              |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Navigation Name    | ``surface_navigation_name``         | ``string``       | ``KSStep``                 | Surface name of the navigation at this step              |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Navigation Flag    | ``surface_navigation_flag``         | ``bool``         | ``KSStep``                 | Additional flag for the surface navigation               |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Initial Particle   | ``initial_particle``                | ``KSParticle``   | ``KSStep``                 | Pointer to initial particle at begin of the step         |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Finale Particle    | ``final_particle``                  | ``KSParticle``   | ``KSStep``                 | Pointer to initial particle at begin of the step         |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Intermed. Particle | ``interaction_particle``            | ``KSParticle``   | ``KSStep``                 | Pointer to initial particle before interaction           |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Intermed. Particle | ``navigation_particle``             | ``KSParticle``   | ``KSStep``                 | Pointer to initial particle before navigation            |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Intermed. Particle | ``terminator_particle``             | ``KSParticle``   | ``KSStep``                 | Pointer to initial particle before termination           |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+
| Intermed. Particle | ``trajectory_particle``             | ``KSParticle``   | ``KSStep``                 | Pointer to initial particle before propagation           |
+--------------------+-------------------------------------+------------------+----------------------------+----------------------------------------------------------+

Vector and matrix type can be accessed by their components in the written output data. For example, when the ``position``
element is used, the corresponding fields in the output data can be found under the names ``position_x``, ``position_y``,
and ``position_z`. For matrix types, the rows are treated as 3-vectors themselves. Hence, the first element in a matrix
field named ``gradient`` can be found under ``gradient_x_x``, and so on.

The following suffixes are available for the vector and matrix types.

+-----------------------------------------------------------------------------+
| Output field suffixes                                                       |
+--------------------+-------------------------------------+------------------+
| Name               | XML Element Suffix                  | Base Type        |
+====================+=====================================+==================+
| X Component        | ``x``                               | ``KThreeVector`` |
+--------------------+-------------------------------------+------------------+
| Y Component        | ``y``                               | ``KThreeVector`` |
+--------------------+-------------------------------------+------------------+
| Z Component        | ``z``                               | ``KThreeVector`` |
+--------------------+-------------------------------------+------------------+
| Vector Magnitude   | ``magnitude``                       | ``KThreeVector`` |
+--------------------+-------------------------------------+------------------+
| Squared Magnitude  | ``magnitude_squared``               | ``KThreeVector`` |
+--------------------+-------------------------------------+------------------+
| Radial Component   | ``perp``                            | ``KThreeVector`` |
+--------------------+-------------------------------------+------------------+
| Squared Radial     | ``perp_squared``                    | ``KThreeVector`` |
+--------------------+-------------------------------------+------------------+
| Polar Angle        | ``polar_angle``                     | ``KThreeVector`` |
+--------------------+-------------------------------------+------------------+
| Azimuthal Angle    | ``azimuthal_angle``                 | ``KThreeVector`` |
+--------------------+-------------------------------------+------------------+
| Determinant        | ``determinant``                     | ``KThreeMatrix`` |
+--------------------+-------------------------------------+------------------+
| Trace              | ``trace``                           | ``KThreeMatrix`` |
+--------------------+-------------------------------------+------------------+


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

