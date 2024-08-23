Tools
*****

Field calculation
=================

Although *Kassiopeia* is quite powerful in terms of configuration options, sometimes it is necessary to calculate the
electric or magnetic field at one or more defined points in the geometry. This is especially useful to compare
different geometry setups, or during the design stage when full-scale simulations are not yet feasible. The field
calculation tools are intended to help with these tasks.

Several programs are available for working with electric fields:

* ``SimpleElectricFieldCalculator`` calculates the electric field and potential at a single point in the global
  coordinate system, and prints the results to the terminal.
* ``SimpleElectricFieldCalculatorAlongZaxis`` calculates the electric field and potential at several points spread
  along the z-axis, defined by a start and stop position on the z-axis and a distance between points. The results are
  printed to the terminal and saved to a output text file.
* ``SimpleElectricFieldCalculatorOverXYplane`` calculates the electric field and potential at several points spread
  over the xy-plane, defined by a position on the z-axis, a maimum radius, and a distance between points. The results
  are printed to the terminal and saved to a output text file.
* ``SimpleElectricFieldCalculatorAlongFieldline`` calculates the electric field and potential along a field line,
  which is calculated with the help of *Kassiopeia* using a magnetic trajectory. In principle this can be done through
  the *Kassiopeia* XML interfaces, but this program provides a convenient method for a simple field line calculation.
  The results are printed to the terminal and saved to a output text file.
* ``SimpleElectricFieldCalculatorFromFile`` takes coordinates from a given input text file and calculates the electric
  field and potential at each point. The results are printed to the terminal and saved to a output text file.

For working with magnetic fields, the same programs are available under the adapted name ``SimpleMagneticFieldCalculator``
and so on. In addition, there is:

* ``SimpleMagneticGradientCalculator`` calculates the magnetic field and its gradient at a single point in the global
  coordinate system, and prints the results to the terminal.

Development
===========

Some KEMField tools are especially useful for development. `InspectEMFile` e.g. prints an overview over all keys in 
`.kbd` files (like KEMField cache files, from which the hash values can be used for the `explicit_superposition_cached_charge_density_solver`). These tools can be found here: :gh-code:`KEMField/Source/Applications/Tools`.

Usage
=====

All listed programs will show a brief usage summary if called without arguments. For example, the
``SimpleElectricFieldCalculatorAlongZaxis`` will show a message:

.. code-block::

    usage: ./SimpleElectricFieldCalculatorAlongZaxis <config_file.xml> <z1> <z2> <dz> <output_file.txt> <electric_field_name1> [<electric_field_name2> <...>]

which indicates that at least 6 arguments are required: the name of a configuration file with at least one defined
electric field; the start and stop position and step distance on the z-axis; the name of an output file; and the name
of an electric field. If multiple fields are specified, their contributions will be summed up.