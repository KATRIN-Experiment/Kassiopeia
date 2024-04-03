Generation & Termination
=========================

Generation
----------

The intial state of particle's to be tracked is set up using the generator mechanism. The abstract base class of all
particle generators is **KSGenerator** and many different implementations exist. When generating a particle,
there are five important initial parameters:

- `PID`: What is the particle type? For particle ID values, see the PDG_ numbering scheme. The PID can also be specified
  by a common name, such as ``e-`` for PID 11 (an electron.)
- `Energy`: What is the initial energy of the particle? The energy is usually specified in Electronvolts (eV).
- `Position`: What is the initial position of the particle?
- `Direction`: In what direction is the particle traveling?
- `Time`: How is the production of particles distributed in time during the simulation?

Each of the dynamic components (energy, position, direction, time) can be draw from a selected probability distribution.
In some scenarios a dedicated particle generator may be need which produces with a very specific and well defined
particle state as the result of some physical process (e.g. electron shake-off in Radon decay). However, as is often the
case, the user may wish to modify each dynamic component in a specific way in order to see what effect this has on the
rest of the simulation.

To draw each dynamic component from an independent distribution a composite generator is used. This type of generator
combines a set of user selected distributions to produce the initial energy, position, direction, and time parameters.
The following composite generator example is taken from the ``DipoleTrapSimulation.xml`` file:

.. code-block:: xml

    <!-- pid=11 implies that electrons will be generated -->
    <ksgen_generator_composite name="generator_uniform" pid="11">
        <energy_composite>
            <energy_fix value="1."/>
        </energy_composite>
        <position_cylindrical_composite surface="world/dipole_trap/center">
            <r_cylindrical radius_min="0." radius_max="2.0e-1"/>
            <phi_uniform value_min="0." value_max="360."/>
            <z_fix value="0."/>
        </position_cylindrical_composite>
        <direction_spherical_composite surface="world/dipole_trap/center">
            <theta_fix value="0."/>
            <phi_uniform value_min="0." value_max="360"/>
        </direction_spherical_composite>
        <time_composite>
            <time_fix value="0."/>
        </time_composite>
    </ksgen_generator_composite>

In this example of the composite generator, the initial kinetic energy of the electron is fixed to 1 eV and its position
is drawn uniformly within a cylindrical volume, defined by the parameters ``(r,phi,z)``.Its initial starting time is
fixed to zero, while its initial momentum direction is fixed along the z-axis by specifiying the corresponding angles
``(phi,theta)`` in a spherical distribution. Here the particle type is specified by the PID 11. The available particles
and their PIDs are defined at the end of the file :gh-code:`Kassiopeia/Operators/Source/KSParticleFactory.cxx`.

Choosing energy values
~~~~~~~~~~~~~~~~~~~~~~

All of the fixed values used in this composite generator may be replaced by probability distributions. The available
probability distributions depend on the quantity they are intended to generate, but include uniform, gaussian, pareto,
cosine, etc. The available distributions can be found in :gh-code:`Kassiopeia/Generators`. Also available is the ability
to generate values at fixed intervals throughout a limited range. For example this can be done for energy as follows:

.. code-block:: xml

        <energy_composite>
            <energy_set name="e_set" value_start="1" value_stop="10" value_count="3"/>
        </energy_composite>

which would generate 3 particles with energies equally spaced between 1 and 10 eV. Alternatively, as specific list of
values can also be used:

.. code-block:: xml

        <energy_composite>
            <energy_list
               add_value="11.8"
               add_value="20.5"
               add_value="33.1"
            />
        </energy_composite>

Keep in mind that if a ``list`` of ``set`` is used within a composite generator, the number of particles
produced in one generation event will be equal to multiplicative combination of all possible particle states.
For example, the following generator specification:

.. code-block:: xml

    <ksgen_generator_composite name="generator_uniform" pid="11">
        <energy_composite>
            <energy_set name="e_set" value_start="1" value_stop="200" value_count="10"/>
        </energy_composite>
        <position_cylindrical_composite surface="world/dipole_trap/center">
            <r_cylindrical radius_min="0." radius_max="2.0e-1"/>
            <phi_uniform value_min="0." value_max="360."/>
            <z_fix value="0."/>
        </position_cylindrical_composite>
        <direction_spherical_composite surface="world/dipole_trap/center">
            <theta_set name="e_set" value_start="0" value_stop="90" values_count="10"/>
            <phi_uniform value_min="0." value_max="360"/>
        </direction_spherical_composite>
        <time_composite>
            <time_fix value="0."/>
        </time_composite>
    </ksgen_generator_composite>

results in a total of 100 particles being generated per event (as a combination of possible energies and momentum
direction theta coordinate). To see other generator examples please see the included example XML files.

The table below lists the available value distributions that can be used with one of the initial parameters. Note
that the XML element name can also be adapted, so instead of ``value_gauss`` for an energy distribution one would use:

.. code-block:: xml

    <energy_composite>
        <energy_gauss mean="18600." sigma="5."/>
    </energy_composite>

Value generator types
~~~~~~~~~~~~~~~~~~~~~

The position and direction generators usually support multiple value distributions; e.g. radius (``r_gauss``),
azimuthal angle (``phi_gauss``) and z-position (``z_gauss``) for the composite cylindrical position generator.

+--------------------------------------------------------------------------------------------------------------------+
| Generator value distributions                                                                                      |
+--------------------+-------------------------------------+---------------------------------------------------------+
| Name               | XML Element                         | Description (main parameters)                           |
+====================+=====================================+=========================================================+
| Fixed              | ``value_fix``                       | Fixed value                                             |
+--------------------+-------------------------------------+---------------------------------------------------------+
| List               | ``value_list``                      | Fixed set of inidivual values                           |
+--------------------+-------------------------------------+---------------------------------------------------------+
| Set                | ``value_set``                       | Fixed set of values in range (start, stop, increment)   |
+--------------------+-------------------------------------+---------------------------------------------------------+
| Uniform            | ``value_uniform``                   | Uniform distribution (min, max)                         |
+--------------------+-------------------------------------+---------------------------------------------------------+
| Boltzmann          | ``value_boltzmann``                 | Boltzmann energy distribution (mass, `kT`)              |
+--------------------+-------------------------------------+---------------------------------------------------------+
| Gauss              | ``value_gauss``                     | Gaussian distribution (mean, sigma, min, max)           |
+--------------------+-------------------------------------+---------------------------------------------------------+
| Generalized Gauss  | ``value_generalized_gauss``         | Skewed Gaussian distrib. (mean, sigma, min, max, skew)  |
+--------------------+-------------------------------------+---------------------------------------------------------+
| Pareto             | ``value_pareto``                    | Pareto distribution (slope, cutoff, offset, min, max)   |
+--------------------+-------------------------------------+---------------------------------------------------------+
| Cylindrical Radius | ``value_radius_cylindrical``        | Cylindrical radial distribution (min, max)              |
+--------------------+-------------------------------------+---------------------------------------------------------+
| Spherical Radius   | ``value_radius_spherical``          | Spherical radial distribution (min, max)                |
+--------------------+-------------------------------------+---------------------------------------------------------+
| Fractional Radius  | ``value_radius_fraction``           | Radial distribution with ``r_max = 1``                  |
+--------------------+-------------------------------------+---------------------------------------------------------+
| Cosine Angle       | ``value_angle_cosine``              | Cosine angular distribution (min, max)                  |
+--------------------+-------------------------------------+---------------------------------------------------------+
| Spherical Angle    | ``value_angle_spherical``           | Spherical angular distribution (min, max)               |
+--------------------+-------------------------------------+---------------------------------------------------------+
| Z-Frustrum         | ``value_z_frustrum``                | Random z-value inside frustrum (z1, r1, z2, r2)         |
+--------------------+-------------------------------------+---------------------------------------------------------+
| Formula            | ``value_formula``                   | ROOT Formula (``TF1``) given as string                  |
+--------------------+-------------------------------------+---------------------------------------------------------+
| Histogram          | ``value_histogram``                 | ROOT Histogram (``TH1``) read from file                 |
+--------------------+-------------------------------------+---------------------------------------------------------+

Special creator types
~~~~~~~~~~~~~~~~~~~~~

In addition, a number of specialized generators exists. For example, the position or energy of the generated particle
can be defined in more a sophisticated way in case a particle is generated from nuclear decays (Tritium, Krypton, Radon)
or starts from a surface.

+--------------------------------------------------------------------------------------------------------------------+
| Energy generators (incomplete list)                                                                                |
+--------------------+-------------------------------------+---------------------------------------------------------+
| Name               | XML Element                         | Description                                             |
+====================+=====================================+=========================================================+
| Beta Decay         | ``energy_beta_decay``               | Energy from (tritium) beta decay                        |
+--------------------+-------------------------------------+---------------------------------------------------------+
| Beta Recoil        | ``energy_beta_recoil``              | Recoil energy from beta decay                           |
+--------------------+-------------------------------------+---------------------------------------------------------+
| Krypton            | ``energy_krypton_event``            | Energy from krypton decay (conversion/Auger)            |
+--------------------+-------------------------------------+---------------------------------------------------------+
| Lead               | ``energy_lead_event``               | Energy from lead decay (conversion/Auger)               |
+--------------------+-------------------------------------+---------------------------------------------------------+
| Radon              | ``energy_radon_event``              | Energy from radon decay (conversion/Auger/ShakeOff)     |
+--------------------+-------------------------------------+---------------------------------------------------------+
| Rydberg            | ``energy_rydberg``                  | Energy from Rydberg ionization                          |
+--------------------+-------------------------------------+---------------------------------------------------------+

+--------------------------------------------------------------------------------------------------------------------+
| Position generators (incomplete list)                                                                              |
+--------------------+-------------------------------------+---------------------------------------------------------+
| Name               | XML Element                         | Description                                             |
+====================+=====================================+=========================================================+
| Cylindrical        | ``position_cylindrical_composite``  | Cylindrical position ``(r, phi, z)``                    |
+--------------------+-------------------------------------+---------------------------------------------------------+
| Rectangular        | ``position_rectangular_composite``  | Rectangular position ``(x, y, z)``                      |
+--------------------+-------------------------------------+---------------------------------------------------------+
| Spherical          | ``position_spherical_composite``    | Spherical position ``(r, phi, theta)``                  |
+--------------------+-------------------------------------+---------------------------------------------------------+
| Flux Tube          | ``position_flux_tube``              | Cylindrical position; radius defined by flux tube       |
+--------------------+-------------------------------------+---------------------------------------------------------+
| Surface            | ``position_surface_random``         | Random position on surface (not all types supported)    |
+--------------------+-------------------------------------+---------------------------------------------------------+
| Mesh Surface       | ``position_mesh_surface_random``    | Random position on surface; needs surface mesh!         |
+--------------------+-------------------------------------+---------------------------------------------------------+
| Space              | ``position_space_random``           | Random position in space (not all types supported)      |
+--------------------+-------------------------------------+---------------------------------------------------------+

Termination
-----------

The converse to particle generation is termination. The abstract base class of all particle terminators is
**KSTerminator**. Terminators are used to stop particle tracks in situations where further simulation of the
particle is of no further interest. Terminators typically operate on very simple conditional logic. For example, a
particle track may be terminated if the particle's kinetic energy drops below some set value, if it intersects a
particular surface, or simply after a given number of steps has been reached.

An example of a terminator which stops particle tracks which exceed the number of allowed steps is given as follows:

.. code-block:: xml

    <ksterm_max_steps name="term_max_steps" steps="1000"/>

A pair of terminators which will terminate a particle that exceeds an allowed range for the z-coordinate is given in the
following example:

.. code-block:: xml

    <ksterm_max_z name="term_max_z" z="1.0"/>
    <ksterm_min_z name="term_min_z" z="-1.0"/>

There are a wide variety of terminators currently avaiable that can be found in :gh-code:`Kassiopeia/Terminators`. The
user is encouraged to peruse the XML example files as well as the source code to determine what (if any) type of
pre-existing terminator might be useful for their purpose. As will be explained later, one may enable/disable specific
terminators dynamically during the simulation. This allows a very flexible configuration of particle termination.


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
